"""Dense inverse-triangulation reconstruction (laser speckle / fringe / correlation).

Pipeline per capture and method
-------------------------------
1. Load and pre-process the stereo images (CLAHE + blur for intensity methods,
   absolute phase + modulation maps for fringe projection).
2. **Coarse pass** – sweep the whole search volume on a 1 mm XY lattice with a
   2 mm depth step and keep, per lattice node, the depth with the best score.
3. **Coarse filtering** – score threshold (loose; the coarse depth step does not
   sample the score peak exactly), image-bounds mask, texture/modulation mask
   and the lattice surface filter (``include/surface_filter.py``).
4. **Fine pass** – refine every lattice node with a 0.1 mm depth step inside a
   +-8 mm window around the smoothed coarse surface (``--fine-search guided``,
   default; the patch follows the seed surface) or re-sweep the whole padded
   bounding box of the coarse surface (``--fine-search full``, the old way).
5. **Fine filtering** – strict score threshold, masks and the lattice surface
   filter again; then a second-chance guided scan that grows the surface into
   nodes lost to false peaks or never seeded (``--fill-holes``).
6. **Fringe phantom rejection** – chance phase matches form smooth surfaces
   that geometry alone cannot reject; they violate the projector light-sheet
   geometry instead (``include/fringe_consistency.py``).
7. Save ``x,y,z,corr`` CSV clouds and a ``*-diagnostics.json`` with the counts
   of every stage.

The outlier rejection used to be a 3-D Euclidean neighbour count (all-pairs
``cdist``). It eroded borders, thinned steep surfaces and still let spikes
through; it is replaced by a local-plane consistency test on the scan lattice.
The old filter remains available with ``--outlier-filter euclidean`` and the
legacy component-based laser refinement with ``--laser-support connected``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

import cv2
import numpy as np
import torch

from include.fringe_projection.include.stereo_fringe_process import FringeProcess
from include.SpatioCorrelation import PyTorchStereoCorrel
from include.laser_surface import supported_components, depth_bands, select_depths, refine_regions
from include.surface_filter import LatticeSurfaceFilter, build_seed_map, lattice_indices, lattice_components, object_groups
from include.fringe_consistency import light_sheet_mask

GRID_STEPS_1 = {'xy': 1.0, 'z': 2.0}   # coarse 3-D patch steps (mm)
GRID_STEPS_2 = {'xy': 1.0, 'z': 0.1}   # fine 3-D patch steps (mm)
FRINGE_PIXELS_PER_PERIOD = 64
FRINGE_STEPS = 8
BLACK_FACTOR = 1.0                     # image scaling before processing (0..1)


# ── Settings ─────────────────────────────────────────────────────────────────

@dataclass
class MethodSettings:
    """Per-method reconstruction parameters.

    Only the score thresholds and the outlier-filter neighbourhood (radius,
    minimum neighbours) are meant to be tuned; kernel, blur and image count
    define the measurement itself.
    """
    method: str
    n_images: int
    kernel: int
    blur: int
    clahe: bool
    threshold: float          # strict score threshold used on the fine pass
    coarse_threshold: float   # loose score threshold used on the coarse pass
    radius: float             # XY neighbourhood radius of the outlier filter (mm)
    min_neighbors: int        # consistent neighbours required by the outlier filter
    mask_bounds: float        # temporal std (intensity methods) or modulation (fringe) floor

    @property
    def lower_is_better(self) -> bool:
        return self.method == 'fringe'

    def score_mask(self, corr: torch.Tensor, threshold: float) -> torch.Tensor:
        finite = torch.isfinite(corr)
        return finite & ((corr < threshold) if self.lower_is_better else (corr > threshold))


# kernel, blur, fine ZNCC threshold. With the surface filter doing the outlier
# rejection the thresholds can be far lower than the 0.65-0.80 used before:
# on the pipe-shell capture the ZNCC peaks of the true surface sit between
# 0.3 and 0.8 (a ~4 px epipolar residual of the calibration depresses them,
# see tools/epipolar_check.py) and the inlier rate after filtering is the same
# for thresholds 0.4-0.6 while the point count roughly doubles.
LASER_PRESETS = {
    'baseline': (3, 1, .50),
    'balanced': (5, 3, .50),
    'precision': (5, 3, .65),
}


def method_settings(method: str, args) -> MethodSettings:
    if method == 'laser':
        kernel, blur, threshold = LASER_PRESETS[args.laser_preset]
        if args.threshold is not None:
            threshold = args.threshold
        # The 2 mm coarse step rarely lands on the ZNCC peak (the peak is only a
        # few mm wide), so the coarse pass is thresholded more loosely and relies
        # on the surface filter to discard the random-depth background.
        coarse = args.coarse_threshold if args.coarse_threshold is not None else min(threshold - 0.05, 0.5)
        return MethodSettings('laser', 5, kernel, blur, True, threshold, coarse,
                              args.radius if args.radius is not None else 10.,
                              args.min_neighbors if args.min_neighbors is not None else 8, 10.)
    if method == 'correl':
        threshold = args.threshold if args.threshold is not None else 0.9
        coarse = args.coarse_threshold if args.coarse_threshold is not None else min(threshold, 0.7)
        return MethodSettings('correl', 15, 1, 1, True, threshold, coarse,
                              args.radius if args.radius is not None else 15.,
                              args.min_neighbors if args.min_neighbors is not None else 8, 20.)
    if method == 'fringe':
        threshold = args.threshold if args.threshold is not None else 0.1
        # |phase_L - phase_R| grows ~0.15 rad per mm of depth error, so with a
        # 2 mm coarse step the nearest sample of a true surface point can sit
        # 0.15 rad away from the minimum.
        coarse = args.coarse_threshold if args.coarse_threshold is not None else max(threshold, 0.3)
        return MethodSettings('fringe', 1, 1, 1, False, threshold, coarse,
                              args.radius if args.radius is not None else 10.,
                              args.min_neighbors if args.min_neighbors is not None else 10,
                              args.fringe_modulation)
    raise ValueError(method)


# ── I/O helpers ───────────────────────────────────────────────────────────────

def save_point_cloud(filename: str | Path, xyz: torch.Tensor, corr: Optional[torch.Tensor] = None, delimiter: str = ','):
    """Save an XYZ cloud, optionally with the per-point score, as CSV."""
    if isinstance(xyz, torch.Tensor):
        xyz = xyz.cpu().numpy()
    if corr is not None and isinstance(corr, torch.Tensor):
        corr = corr.cpu().numpy()
    if corr is not None:
        data = np.hstack((xyz, corr.reshape(-1, 1)))
        header = 'x,y,z,corr'
    else:
        data, header = xyz, 'x,y,z'
    np.savetxt(filename, data, delimiter=delimiter, header=header, comments='')
    print(f'Nuvem de pontos salva em {filename}')


def read_images(path: Path, names: list, n_imgs: int) -> list:
    imgs = [cv2.imread(str(path / n), cv2.IMREAD_GRAYSCALE) for n in names[:n_imgs]]
    if any(i is None for i in imgs):
        raise FileNotFoundError(f'Could not read all images in {path}')
    return [cv2.convertScaleAbs(i, alpha=BLACK_FACTOR) for i in imgs]


def find_images(capture: Path, method: str) -> Optional[Path]:
    """Locate the stereo folders of a capture: ``capture/method/left`` or ``capture/left``."""
    for candidate in (capture / method, capture):
        if (candidate / 'left').is_dir() and (candidate / 'right').is_dir() and any((candidate / 'left').iterdir()):
            return candidate
    return None


def load_method_images(method: str, images_path: Path, n_imgs: int, debug: bool = False):
    """Return (left, right) image lists ready for ``PyTorchStereoCorrel.convert_images``."""
    left_path, right_path = images_path / 'left', images_path / 'right'
    left_names = sorted(p.name for p in left_path.iterdir())
    right_names = sorted(p.name for p in right_path.iterdir())
    if not left_names or not right_names:
        raise FileNotFoundError(f'No images found in {images_path}')
    if method == 'fringe':
        left = read_images(left_path, left_names, len(left_names))
        right = read_images(right_path, right_names, len(right_names))
        proc = FringeProcess(camera_resolution=left[0].shape[::-1], px_f=FRINGE_PIXELS_PER_PERIOD, steps=FRINGE_STEPS)
        proc.images_left = np.moveaxis(np.array(left), 0, -1)
        proc.images_right = np.moveaxis(np.array(right), 0, -1)
        abs_left, abs_right, mod_left, mod_right = proc.calculate_abs_phi_images(visualize=debug)
        return ([abs_left.cpu().numpy(), mod_left.cpu().numpy()],
                [abs_right.cpu().numpy(), mod_right.cpu().numpy()])
    return read_images(left_path, left_names, n_imgs), read_images(right_path, right_names, n_imgs)


# ── Filtering stages ──────────────────────────────────────────────────────────

def apply_masks(zscan: PyTorchStereoCorrel, xyz: torch.Tensor, corr: torch.Tensor, cfg: MethodSettings,
                mask_uv: bool = True, mask_std: bool = True):
    """Image-bounds and texture/modulation masks (unchanged from the original pipeline)."""
    interp = None
    if mask_uv and len(xyz):
        xyz, corr, interp = zscan.mask_uv_points(xyz, corr)
    if mask_std and len(xyz):
        xyz, corr, interp = zscan.std_mask_points(xyz, corr, bounds=cfg.mask_bounds, method=cfg.method)
    return xyz, corr, interp


def outlier_filter(zscan: PyTorchStereoCorrel, args, xyz: torch.Tensor, corr: torch.Tensor, cfg: MethodSettings,
                   z_step: float, interp: Optional[torch.Tensor] = None, stats: Optional[dict] = None, stage: str = ''):
    """Dispatch to the configured outlier filter and log the result."""
    n_in = len(xyz)
    if n_in == 0 or args.outlier_filter == 'none':
        return xyz, corr
    if args.outlier_filter == 'euclidean':
        fn = zscan.euclidean_filter if args.neighbor_backend == 'torch' else zscan.euclidean_filter_kdtree
        xyz, corr, _ = fn(xyz_gpu=xyz, corr_gpu=corr, interp=interp, min_neighbors=cfg.min_neighbors, radius=cfg.radius)
        print(f'{stage} euclidean filter: {n_in} -> {len(xyz)} points')
        if stats is not None:
            stats[f'{stage}_filter'] = dict(kind='euclidean', input=n_in, kept=len(xyz))
        return xyz, corr
    flt = LatticeSurfaceFilter(radius=cfg.radius, min_neighbors=cfg.min_neighbors, tolerance=args.plane_tolerance,
                               z_step=z_step, max_slope=args.max_slope)
    keep = flt.mask(xyz, zscan.x_vals, zscan.y_vals)
    print(f'{stage} {flt.report}')
    if stats is not None:
        stats[f'{stage}_filter'] = dict(kind='lattice', **asdict(flt.report))
    return xyz[keep], corr[keep]


def light_sheet_stage(zscan, args, xyz, corr, cfg, floor_rad, stats, stage):
    """Fringe only: reject chance phase matches that violate the projector geometry."""
    if cfg.method != 'fringe' or args.no_light_sheet_check or len(xyz) == 0:
        return xyz, corr
    keep, report = light_sheet_mask(zscan, xyz, floor_rad=floor_rad)
    print(f'{stage} {report}')
    if stats is not None:
        stats[f'{stage}_light_sheet'] = asdict(report)
    return xyz[keep], corr[keep]


def distinct_peak(cfg, args, score, runner_up):
    """Laser/correl: the winning ZNCC peak must stand above the rest of the window.

    ``runner_up`` is the best score at least 2 mm away from the winner. A true
    correspondence has a single clear peak, a chance match in textureless or
    occluded regions has several of similar height; requiring
    ``score - runner_up >= --min-prominence`` rejects the latter regardless of
    the absolute score level.
    """
    if cfg.lower_is_better or args.min_prominence <= 0:
        return torch.ones_like(score, dtype=torch.bool)
    return (score - runner_up) >= args.min_prominence


def component_stage(zscan, args, xyz, corr, stats):
    """Drop small depth-continuous components (clusters 'lost in space').

    Points are connected when within 3 mm in XY and the depth slope between
    them is below 3 mm/mm, so sparse but genuine surface stays connected while
    detached clusters (chance matches on the background, image-border
    artefacts) form their own small components.
    """
    if len(xyz) == 0:
        return xyz, corr
    if args.min_component > 0:
        labels, sizes = lattice_components(xyz, zscan.x_vals, zscan.y_vals, jump=3.0, reach=3.0)
        keep = (labels >= 0) & (sizes[labels.clamp_min(0)] >= args.min_component)
        n_small = int((sizes < args.min_component).sum())
        print(f'component filter: {len(xyz)} -> {int(keep.sum())} points ({len(sizes)} components, '
              f'{n_small} smaller than {args.min_component} nodes removed)')
        if stats is not None:
            stats['components'] = dict(input=len(xyz), kept=int(keep.sum()), components=len(sizes), removed_small=n_small)
        xyz, corr = xyz[keep], corr[keep]
    if args.keep_groups > 0 and len(xyz):
        # Objects: points within --group-distance mm (3-D) of each other, so a
        # part separated by a step or a small gap stays with its object while
        # background surfaces lit by the projector form their own groups.
        labels, sizes = object_groups(xyz, args.group_distance)
        order = torch.argsort(sizes, descending=True)
        chosen = order[:args.keep_groups]
        keep = (labels >= 0) & torch.isin(labels, chosen)
        print(f'object groups ({args.group_distance:g} mm): sizes {sizes[order[:6]].tolist()}; '
              f'keeping the {args.keep_groups} largest -> {int(keep.sum())} points')
        if stats is not None:
            stats['groups'] = dict(sizes=sizes[order[:10]].tolist(), kept=int(keep.sum()))
        xyz, corr = xyz[keep], corr[keep]
    return xyz, corr


def filter_stage(zscan, args, xyz, corr, cfg, threshold, z_step, stats, stage, edge=None):
    """Threshold -> masks -> outlier filter, returning the surviving cloud.

    ``edge`` marks nodes whose winner sits on the first/last depth sample of the
    sweep: the real extremum is outside the range, so they are discarded.
    """
    mask = cfg.score_mask(corr, threshold)
    if edge is not None:
        mask &= ~edge
    xyz, corr = xyz[mask], corr[mask]
    label = f'< {threshold} rad' if cfg.lower_is_better else f'> {threshold:.2f}'
    print(f'{stage} score {label}: {len(xyz)} points')
    xyz, corr, interp = apply_masks(zscan, xyz, corr, cfg, not args.no_uv_mask, not args.no_std_mask)
    print(f'{stage} after image/texture masks: {len(xyz)} points')
    if stats is not None:
        stats[f'{stage}_after_threshold'] = int(mask.sum())
        stats[f'{stage}_after_masks'] = len(xyz)
    xyz, corr = outlier_filter(zscan, args, xyz, corr, cfg, z_step, interp, stats, stage)
    return xyz, corr


# ── Reconstruction of one capture/method ─────────────────────────────────────

def reconstruct(zscan: PyTorchStereoCorrel, args, obj: str, method: str, images_path: Path, output_path: Path):
    cfg = method_settings(method, args)
    limits = {'x': tuple(args.x_limits), 'y': tuple(args.y_limits), 'z': tuple(args.z_limits)}
    stats: dict = dict(object=obj, method=method, settings=asdict(cfg), fine_search=args.fine_search,
                       outlier_filter=args.outlier_filter)
    use_surface_support = method == 'laser' and args.laser_support == 'connected'
    validate_split = use_surface_support and args.laser_split_tolerance >= 0
    t0 = time.time()

    print(f'Carregando imagens de {images_path} ...')
    left, right = load_method_images(method, images_path, cfg.n_images, debug=args.debug)
    if method == 'laser':
        print(f'Laser preset: {args.laser_preset}, world patch {cfg.kernel}x{cfg.kernel}, image blur {cfg.blur}x{cfg.blur}')
    print(f'Iniciando {method} com kernel espacial {cfg.kernel}x{cfg.kernel}')
    zscan.convert_images(left, right, apply_clahe=cfg.clahe, undist=False, apply_blur=cfg.blur > 1, blur=cfg.blur)
    del left, right
    kernel = cfg.kernel
    # Keep whole patches inside the images: half kernel (mm) at <= 6 px/mm plus
    # the bicubic/blur support. Patches hanging over the border see replicated
    # edge pixels in both views and correlate spuriously.
    zscan.image_margin = args.image_margin if args.image_margin >= 0 else 3 + int(np.ceil(6 * (kernel // 2) * GRID_STEPS_1['xy']))
    print(f'Image border margin: {zscan.image_margin} px')
    if cfg.method != 'fringe':
        print(f'Background temporal noise level: {zscan.noise_level:.2f} -> texture mask bound '
              f'{min(cfg.mask_bounds, max(2.0, 3.0 * zscan.noise_level)):.2f}')
    suffix = f'-{args.laser_preset}' if method == 'laser' and args.laser_preset != 'baseline' else ''
    if use_surface_support:
        suffix += '-supported'
    cloud_file = output_path / f'{obj}-{method}-{cfg.n_images}img-{kernel}kernel{suffix}.csv'
    print(f'Pré-processamento concluído em {time.time() - t0:.2f} s')

    # ── 1. Coarse pass ───────────────────────────────────────────────────────
    t1 = time.time()
    zscan.points3d(x_lim=limits['x'], y_lim=limits['y'], z_lim=limits['z'],
                   xy_step=GRID_STEPS_1['xy'], z_step=GRID_STEPS_1['z'])
    if method == 'laser' and args.surface_seeds:
        seed_files = sorted(args.surface_seeds.glob(f'{obj}-laser*.csv'))
        if len(seed_files) != 1:
            raise ValueError(f'Expected one laser seed cloud for {obj}, found {len(seed_files)}')
        seeds = np.loadtxt(seed_files[0], delimiter=',', skiprows=1, ndmin=2)
        xyz = torch.as_tensor(seeds[:, :3], dtype=torch.float32, device=zscan.device)
        corr = torch.full((len(xyz),), float('inf'), dtype=torch.float32, device=zscan.device)
        print(f'Reusing {len(xyz)} geometry seeds from {seed_files[0]}; coarse search skipped')
    else:
        cache = None
        if args.coarse_cache is not None:
            args.coarse_cache.mkdir(parents=True, exist_ok=True)
            tag = '_'.join(f'{v:g}' for v in (*limits['x'], *limits['y'], *limits['z']))
            cache = args.coarse_cache / f'{obj}-{method}-{kernel}k-{cfg.blur}b-{cfg.n_images}img-{tag}.pt'
        if cache is not None and cache.exists():
            cached = torch.load(cache, map_location=zscan.device)
            xyz, corr, z_idx = cached['xyz'], cached['corr'], cached['z_idx']
            print(f'Coarse pass loaded from cache {cache}')
        else:
            xyz, corr, z_idx = zscan.process_segmented_z(Kx=kernel, Ky=kernel, stride=1, Nz_block_voxels=10, method=method)
            if cache is not None:
                torch.save({'xyz': xyz.cpu(), 'corr': corr.cpu(), 'z_idx': z_idx.cpu()}, cache)
        edge = (z_idx <= 0) | (z_idx >= len(zscan.z_vals) - 1)
    stats['coarse_seconds'] = time.time() - t1
    print(f'1a Correlação concluída em {stats["coarse_seconds"]:.2f} s; {len(xyz)} nós avaliados')
    if xyz.numel() == 0:
        print('Nenhum ponto retornado pelo processamento grosseiro.')
        return

    if method == 'laser' and args.surface_seeds:
        finite = torch.isfinite(xyz).all(1)
        xyz, corr = xyz[finite], corr[finite]
    else:
        xyz, corr = filter_stage(zscan, args, xyz, corr, cfg, cfg.coarse_threshold, GRID_STEPS_1['z'], stats,
                                 'coarse', edge=edge)
        # Coarse depths are quantised to +-1 mm, i.e. a few tenths of a radian
        # of phase: use a generous floor here, the tight one after refinement.
        xyz, corr = light_sheet_stage(zscan, args, xyz, corr, cfg, 2.0, stats, 'coarse')
    if args.debug:
        zscan.plot_3d_points(xyz[:, 0], xyz[:, 1], xyz[:, 2], corr, title=f'coarse {method} {obj}')

    bands = None
    if use_surface_support:
        before = len(xyz)
        support_mask, components = supported_components(xyz.cpu().numpy(), radius=args.surface_radius,
                                                        min_points=args.surface_min_points)
        support_mask = torch.as_tensor(support_mask, device=zscan.device)
        xyz, corr = xyz[support_mask], corr[support_mask]
        bands = depth_bands(components, margin=args.depth_band_margin, limits=limits['z'])
        stats.update(coarse_before_support=before, coarse_supported=len(xyz), components=components, depth_bands=bands)
        print(f'Coarse surface support: {before} -> {len(xyz)} points, {len(components)} components')

    if xyz.numel() == 0:
        print('No points survived coarse filtering; skipping refinement.')
        save_point_cloud(cloud_file, xyz, corr)
        return

    # ── 2. Fine lattice around the coarse surface ────────────────────────────
    pad = (kernel // 2 + args.fine_padding) * GRID_STEPS_2['xy']
    xlim = (max(limits['x'][0], float(xyz[:, 0].min()) - pad), min(limits['x'][1], float(xyz[:, 0].max()) + pad))
    ylim = (max(limits['y'][0], float(xyz[:, 1].min()) - pad), min(limits['y'][1], float(xyz[:, 1].max()) + pad))
    z_margin = args.fine_window if args.fine_window > 0 else 4 * GRID_STEPS_1['z']
    zlim = (max(limits['z'][0], float(xyz[:, 2].min()) - z_margin), min(limits['z'][1], float(xyz[:, 2].max()) + z_margin))
    if use_surface_support:
        zlim = (bands[0][0], bands[-1][1])
    coarse_xyz = xyz
    zscan.points3d(x_lim=xlim, y_lim=ylim, z_lim=zlim, xy_step=GRID_STEPS_2['xy'], z_step=GRID_STEPS_2['z'])
    if args.fine_search == 'guided' and not use_surface_support:
        print(f'Fine lattice: X {xlim[0]:.0f}..{xlim[1]:.0f}, Y {ylim[0]:.0f}..{ylim[1]:.0f} '
              f'({len(zscan.x_vals)}x{len(zscan.y_vals)} nodes, depth window per node)')
    else:
        print(f'Fine lattice: X {xlim[0]:.0f}..{xlim[1]:.0f}, Y {ylim[0]:.0f}..{ylim[1]:.0f}, '
              f'Z {zlim[0]:.1f}..{zlim[1]:.1f} ({len(zscan.x_vals)}x{len(zscan.y_vals)}x{len(zscan.z_vals)} voxels)')
    stats['fine_lattice'] = [len(zscan.x_vals), len(zscan.y_vals), len(zscan.z_vals)]

    # ── 3. Fine pass ─────────────────────────────────────────────────────────
    t2 = time.time()
    seed = None
    edge = None
    if use_surface_support:
        depth_mask = select_depths(zscan.z_vals.cpu().numpy(), bands)
        zscan.z_vals = zscan.z_vals[torch.as_tensor(depth_mask, device=zscan.device)]
        xyz, corr, region_stats = refine_regions(zscan, components, kernel, xy_margin=GRID_STEPS_1['xy'] * 3,
                                                 z_margin=args.depth_band_margin, validate_split=validate_split,
                                                 seed_points=coarse_xyz.cpu().numpy(), tile_size=args.refine_tile_size)
        stats.update(region_stats)
    elif args.fine_search == 'guided':
        seed = build_seed_map(coarse_xyz, zscan.x_vals, zscan.y_vals, fill_radius=args.seed_fill_radius,
                              smooth_radius=GRID_STEPS_1['z'])
        half = args.fine_window if args.fine_window > 0 else 4 * GRID_STEPS_1['z']
        print(f'Guided fine pass: +-{half:.1f} mm around the coarse surface '
              f'({int(torch.isfinite(seed).sum())} seeded nodes)')
        xyz, corr, k_idx, runner_up = zscan.process_guided_z(kernel, kernel, seed, half_window=half,
                                                             z_step=GRID_STEPS_2['z'], Nz_block_voxels=10, method=method)
        n_off = int(round(half / GRID_STEPS_2['z']))
        edge = k_idx.abs() >= n_off
        if args.save_raw and len(xyz):
            save_point_cloud(cloud_file.with_name(cloud_file.stem + '-raw-runnerup.csv'), xyz, runner_up)
        edge |= ~distinct_peak(cfg, args, corr, runner_up)
    else:
        xyz, corr, z_idx = zscan.process_segmented_z(Kx=kernel, Ky=kernel, stride=1, Nz_block_voxels=10, method=method)
        edge = (z_idx <= 0) | (z_idx >= len(zscan.z_vals) - 1)
    stats['fine_seconds'] = time.time() - t2
    print(f'2a Correlação concluída em {stats["fine_seconds"]:.2f} s; {len(xyz)} nós avaliados')
    if args.save_raw and len(xyz):
        save_point_cloud(cloud_file.with_name(cloud_file.stem + '-raw.csv'), xyz, corr)
    if xyz.numel() == 0:
        print('Nenhum ponto retornado pelo refinamento.')
        save_point_cloud(cloud_file, xyz, corr)
        return

    # ── 4. Fine filtering ────────────────────────────────────────────────────
    if validate_split and zscan.laser_diagnostics is not None:
        split_ok = zscan.laser_diagnostics['split_depth_error'] <= args.laser_split_tolerance
        print(f'Fine frame-split consistency: {len(xyz)} -> {int(split_ok.sum())}')
        xyz, corr = xyz[split_ok], corr[split_ok]
        edge = None
    xyz, corr = filter_stage(zscan, args, xyz, corr, cfg, cfg.threshold, GRID_STEPS_2['z'], stats, 'fine',
                             edge=edge if not use_surface_support else None)

    # ── 5. Second chance for nodes lost to false peaks ───────────────────────
    if args.fill_holes and not use_surface_support and len(xyz):
        t3 = time.time()
        xyz, corr = fill_holes(zscan, args, xyz, corr, cfg, kernel, stats)
        stats['fill_seconds'] = time.time() - t3

    xyz, corr = light_sheet_stage(zscan, args, xyz, corr, cfg, args.light_sheet_floor, stats, 'final')
    xyz, corr = component_stage(zscan, args, xyz, corr, stats)

    if use_surface_support and len(xyz):
        support_mask, final_components = supported_components(xyz.cpu().numpy(), radius=args.surface_radius,
                                                              min_points=args.surface_min_points)
        support_mask = torch.as_tensor(support_mask, device=zscan.device)
        xyz, corr = xyz[support_mask], corr[support_mask]
        stats.update(final_components=final_components)

    stats.update(final_points=len(xyz), total_seconds=time.time() - t0)
    save_point_cloud(cloud_file, xyz, corr)
    (output_path / f'{obj}-{method}-diagnostics.json').write_text(json.dumps(stats, indent=2, default=str))
    print(f'======== Concluído: {len(xyz)} pontos em {stats["total_seconds"]:.2f} s ========')
    if not args.no_plot:
        zscan.plot_3d_points(xyz[:, 0], xyz[:, 1], xyz[:, 2], corr,
                             title=f'{method} - {obj}: {cfg.n_images} imgs {kernel}x{kernel}')


def fill_holes(zscan, args, xyz, corr, cfg, kernel, stats):
    """Grow the accepted surface into unaccepted lattice nodes (second chance).

    In the fine sweep a node whose true score peak is weak can be won by a
    random peak elsewhere in the depth window; it is then (correctly) rejected
    by the surface filter, leaving a hole. Nodes outside the coarse seed reach
    are never evaluated at all. Here the surface interpolated/extrapolated
    from the accepted points (``build_seed_map``) provides a depth seed for
    those nodes, which are re-scanned in a narrow window (``--fill-window``)
    and accepted under the *same* rules as everything else: score threshold,
    image/texture masks and lattice surface consistency. Because each round
    extends the reach by up to ``--seed-fill-radius`` mm, the loop repeats
    until growth stops or ``--fill-iterations`` is reached.
    """
    nx, ny = len(zscan.x_vals), len(zscan.y_vals)
    half = max(4 * GRID_STEPS_2['z'], min(args.fill_window, args.seed_fill_radius))
    n_off = int(round(half / GRID_STEPS_2['z']))
    rounds = []
    for it in range(args.fill_iterations):
        ix, iy, _ = lattice_indices(xyz, zscan.x_vals, zscan.y_vals)
        accepted = torch.zeros((nx, ny), dtype=torch.bool, device=zscan.device)
        accepted[ix, iy] = True
        seed = build_seed_map(xyz, zscan.x_vals, zscan.y_vals, fill_radius=args.seed_fill_radius)
        candidates = torch.isfinite(seed) & ~accepted
        n_cand = int(candidates.sum())
        if n_cand == 0:
            break
        new_xyz, new_corr, k_idx, runner_up = zscan.process_guided_z(kernel, kernel, seed, half_window=half,
                                                                     z_step=GRID_STEPS_2['z'], Nz_block_voxels=10,
                                                                     method=cfg.method, node_mask=candidates)
        mask = cfg.score_mask(new_corr, cfg.threshold) & (k_idx.abs() < n_off) & distinct_peak(cfg, args, new_corr, runner_up)
        new_xyz, new_corr = new_xyz[mask], new_corr[mask]
        new_xyz, new_corr, _ = apply_masks(zscan, new_xyz, new_corr, cfg, not args.no_uv_mask, not args.no_std_mask)
        merged_xyz = torch.cat([xyz, new_xyz])
        merged_corr = torch.cat([corr, new_corr])
        flt = LatticeSurfaceFilter(radius=cfg.radius, min_neighbors=cfg.min_neighbors,
                                   tolerance=args.plane_tolerance, z_step=GRID_STEPS_2['z'], max_slope=args.max_slope)
        keep = flt.mask(merged_xyz, zscan.x_vals, zscan.y_vals)
        keep[:len(xyz)] = True   # never drop already accepted points because of new candidates
        added = int(keep[len(xyz):].sum())
        print(f'Hole filling round {it + 1}: {n_cand} candidate nodes, {len(new_xyz)} passed thresholds, {added} accepted')
        rounds.append(dict(candidates=n_cand, passed_threshold=len(new_xyz), accepted=added))
        xyz, corr = merged_xyz[keep], merged_corr[keep]
        if added < max(10, n_cand // 200):
            break
    stats['fill_holes'] = rounds
    return xyz, corr


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Dense inverse-triangulation reconstruction',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    root = Path(__file__).resolve().parent
    parser.add_argument('--calib', type=Path, default=root / 'cfg' / 'calib.yaml')
    parser.add_argument('--image-root', type=Path, default=root / 'data')
    parser.add_argument('--output', type=Path, default=root / 'reconstruction')
    parser.add_argument('--objects', nargs='+', help='Capture folder names; defaults to all captures')
    parser.add_argument('--methods', nargs='+', choices=['laser', 'fringe', 'correl'], default=['laser'])
    parser.add_argument('--x-limits', nargs=2, type=float, default=(-200., 600.), metavar=('MIN', 'MAX'))
    parser.add_argument('--y-limits', nargs=2, type=float, default=(-200., 600.), metavar=('MIN', 'MAX'))
    parser.add_argument('--z-limits', nargs=2, type=float, default=(-800., 800.), metavar=('MIN', 'MAX'))
    parser.add_argument('--laser-preset', choices=list(LASER_PRESETS), default='balanced',
                        help='baseline: 3x3, corr>.50; balanced: 5x5 + blur3, corr>.50; precision: 5x5 + blur3, corr>.65')
    parser.add_argument('--fringe-modulation', type=float, default=10.,
                        help='Minimum fringe modulation at every interpolation support pixel')
    # Tunable acceptance parameters
    parser.add_argument('--threshold', type=float, help='Fine-pass score threshold (overrides the method default)')
    parser.add_argument('--coarse-threshold', type=float, help='Coarse-pass score threshold (default: looser than fine)')
    parser.add_argument('--radius', type=float, help='Outlier-filter XY neighbourhood radius in mm')
    parser.add_argument('--min-neighbors', type=int, help='Outlier-filter minimum consistent neighbours')
    parser.add_argument('--plane-tolerance', type=float,
                        help='Lattice filter slab half-thickness in mm (default: radius/10, floored at 2 depth steps)')
    # Pipeline structure
    parser.add_argument('--outlier-filter', choices=['lattice', 'euclidean', 'none'], default='lattice')
    parser.add_argument('--neighbor-backend', choices=['torch', 'kdtree'], default='torch',
                        help='Backend of the legacy Euclidean filter')
    parser.add_argument('--fine-search', choices=['full', 'guided'], default='guided',
                        help='full: sweep the whole depth range of the coarse box; guided: window around the coarse surface')
    parser.add_argument('--fine-window', type=float, default=0.,
                        help='Half width (mm) of the guided window / depth margin of the full box; 0 = automatic')
    parser.add_argument('--fine-padding', type=int, default=6, help='Extra lattice cells around the coarse XY box')
    parser.add_argument('--fill-holes', dest='fill_holes', action='store_true', default=True,
                        help='Second-chance guided scan of nodes rejected in the fine pass')
    parser.add_argument('--no-fill-holes', dest='fill_holes', action='store_false')
    parser.add_argument('--fill-window', type=float, default=3., help='Half width (mm) of the hole-filling window')
    parser.add_argument('--fill-iterations', type=int, default=12, help='Maximum hole-filling / growth rounds')
    parser.add_argument('--seed-fill-radius', type=float, default=12.,
                        help='How far (mm) the seed surface is interpolated/extrapolated from accepted points')
    parser.add_argument('--no-light-sheet-check', action='store_true',
                        help='Fringe: skip the projector light-sheet consistency test that removes phantom matches')
    parser.add_argument('--light-sheet-floor', type=float, default=0.3,
                        help='Fringe: minimum phase tolerance (rad) of the final light-sheet test')
    parser.add_argument('--max-slope', type=float, default=3.,
                        help='Maximum local surface slope |dz/dxy| accepted by the lattice filter (mm/mm, 0 = no limit)')
    parser.add_argument('--min-prominence', type=float, default=0.15,
                        help='Laser/correl: minimum ZNCC margin of the winner over the best peak >= 2 mm away (0 = off)')
    parser.add_argument('--image-margin', type=int, default=-1,
                        help='Pixels a projected node must keep from the image border (-1 = automatic from the kernel)')
    parser.add_argument('--min-component', type=int, default=100,
                        help='Drop depth-continuous lattice components smaller than this many nodes (0 = keep all)')
    parser.add_argument('--keep-groups', type=int, default=2,
                        help='Keep only the N largest 3-D-connected object groups (0 = keep all, e.g. background surfaces)')
    parser.add_argument('--group-distance', type=float, default=20.,
                        help='3-D distance (mm) that connects points into the same object group')
    parser.add_argument('--no-uv-mask', action='store_true')
    parser.add_argument('--no-std-mask', action='store_true')
    parser.add_argument('--save-raw', action='store_true', help='Also save the unfiltered fine-pass cloud')
    parser.add_argument('--coarse-cache', type=Path,
                        help='Directory used to cache/reuse coarse-pass results (same images, kernel and limits)')
    parser.add_argument('--no-plot', action='store_true', help='Run reconstruction without interactive plots')
    parser.add_argument('--debug', action='store_true', help='Show intermediate plots')
    # Legacy laser component refinement
    parser.add_argument('--laser-support', choices=['connected', 'off'], default='off',
                        help='connected: legacy component-based laser refinement (superseded by the lattice filter)')
    parser.add_argument('--surface-radius', type=float, default=5., help='3D connectivity radius in mm (legacy)')
    parser.add_argument('--surface-min-points', type=int, default=16, help='Minimum supported component size (legacy)')
    parser.add_argument('--depth-band-margin', type=float, default=10., help='Padding around component depth bands (legacy)')
    parser.add_argument('--laser-split-tolerance', type=float, default=1.,
                        help='Maximum fine-pass frame-split depth difference in mm; negative disables (legacy)')
    parser.add_argument('--surface-seeds', type=Path,
                        help='Reuse existing laser clouds of the same captures as geometry seeds, skipping the coarse scan')
    parser.add_argument('--refine-tile-size', type=float, default=32.,
                        help='Fine refinement neighborhoods in mm; zero uses whole component boxes (legacy)')
    return parser.parse_args(argv)


def main(args=None):
    args = parse_args() if args is None else args
    zscan = PyTorchStereoCorrel(yaml_file=str(args.calib))
    zscan.fringe_min_modulation = args.fringe_modulation
    objects = args.objects or sorted(p.name for p in args.image_root.iterdir() if p.is_dir())
    output_path = args.output
    output_path.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    for obj in objects:
        for method in args.methods:
            print('=' * 50)
            print(f'Iniciando processamento para {obj} usando o método {method}...')
            images_path = find_images(args.image_root / obj, method)
            if images_path is None:
                print(f"Skipping '{args.image_root / obj}': no '{method}/left' or 'left' folder with images.")
                continue
            try:
                reconstruct(zscan, args, obj, method, images_path, output_path)
            except FileNotFoundError as exc:
                print(f'Erro: {exc}')
            if zscan.device.type == 'cuda':
                torch.cuda.empty_cache()
    print(f'\nProcessamento total concluído em {time.time() - t_start:.2f} s.')
    print(f'Resultados salvos em: {output_path.resolve()}')


if __name__ == '__main__':
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f'GPU detectada: {props.name}, Memória Total: {props.total_memory / (1024 ** 2):.2f} MB')
    else:
        print('GPU não detectada pelo PyTorch. O código será executado na CPU, o que pode ser muito lento.')
    main()