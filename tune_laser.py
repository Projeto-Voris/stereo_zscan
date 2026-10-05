"""Sampled inverse-triangulation sweep using the calibration and captures of zscan.py.

No image downsampling, no change to world-patch spacing, and no use of saved
depths to restrict the global search. Saved clouds only supply XY sample sites
and a regression reference. Frame-split stability is a proxy, not accuracy.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import time

os.environ.setdefault('MPLCONFIGDIR', '/tmp/stereo_zscan_mpl')
import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree
from include.SpatioCorrelation import PyTorchStereoCorrel

ROOT = Path('/home/daniel/Pictures/20261001 - Avaliação reconstrução sistemas ar/active_stereo')
VARIANTS = [('baseline', True, 2., 1, 3), ('raw_k3', False, 2., 1, 3),
            ('clahe1_k3', True, 1., 1, 3), ('blur3_k3', True, 2., 3, 3),
            ('clahe2_k1', True, 2., 1, 1), ('clahe2_k5', True, 2., 1, 5),
            ('clahe2_k7', True, 2., 1, 7), ('blur3_k5', True, 2., 3, 5),
            ('raw_k5', False, 2., 1, 5), ('temporal_k3', True, 2., 1, 3),
            ('temporal_k5', True, 2., 1, 5)]


def patches(model, xy, depths, k):
    """Return B,N,K*K,T samples on a 1 mm world XY lattice."""
    b, n = depths.shape
    di, dj = torch.meshgrid(torch.arange(-(k//2), k//2+1, device=model.device),
                           torch.arange(-(k//2), k//2+1, device=model.device), indexing='ij')
    offsets = torch.stack([di.ravel(), dj.ravel()], 1)
    pxy = xy[:, None, :] + offsets[None, :, :]
    pts = torch.cat([pxy[None].expand(b, -1, -1, -1),
                     depths[:, :, None, None].expand(-1, -1, k*k, 1)], -1).reshape(-1, 3)
    result = []
    for cam, images in [('left', model.left_images), ('right', model.right_images)]:
        uv, mask = model.transform_gcs2ccs(pts, cam, image_shape=images.shape[1:])
        result.append(model.interpolate_images(images, uv, mask).reshape(b, n, k*k, -1))
    return result


def score(l, r, k, frames, temporal=False):
    a, b = l[:, :, :, frames], r[:, :, :, frames]
    if temporal:
        a, b = a-a.mean(-1, keepdim=True), b-b.mean(-1, keepdim=True)
    a, b = a.flatten(2), b.flatten(2)
    # Same raw-moment ZNCC as the original implementation.
    count = a.shape[-1]
    sa, sb = a.sum(-1), b.sum(-1)
    numerator = (a*b).sum(-1) - sa*sb/count
    va, vb = (a*a).sum(-1)-sa*sa/count, (b*b).sum(-1)-sb*sb/count
    return torch.nan_to_num(numerator/(va*vb).sqrt().clamp_min(1e-8), nan=-torch.inf)


@torch.inference_mode()
def search(model, xy, k, frames, seeds=None, dz=2.):
    if seeds is None:
        levels = torch.arange(-800., 800.+dz/2, dz, device=model.device)
        depths = levels[:, None].expand(-1, len(xy))
    else:
        levels = torch.arange(-2., 2.+dz/2, dz, device=model.device)
        depths = (seeds[None, :] + levels[:, None]).clamp(-800., 800.)
    scores = []
    for start in range(0, len(levels), 32):
        l, r = patches(model, xy, depths[start:start+32], k)
        scores.append(score(l, r, k, frames, getattr(model, 'tuning_temporal', False)))
    scores = torch.cat(scores)
    best, index = scores.max(0)
    z = depths.gather(0, index[None]).squeeze(0)
    if seeds is None:
        far = (depths-z[None]).abs() >= 10.
        competitor = scores.masked_fill(~far, -torch.inf).max(0).values
        return z, best, best-competitor
    return z, best, None


def finite_median(a):
    a = np.asarray(a)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if len(a) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=Path('laser_tuning'))
    parser.add_argument('--samples', type=int, default=96)
    parser.add_argument('--captures', nargs='*')
    parser.add_argument('--variants', nargs='*', choices=[v[0] for v in VARIANTS])
    parser.add_argument('--global-fine', action='store_true',
                        help='Search every depth at 0.1 mm; slower, avoids coarse-peak aliasing')
    parser.add_argument('--sites-from', type=Path, help='Reuse XY samples from an earlier sweep directory')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.samples < 2:
        parser.error('--samples must be at least 2')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(args.threads)
    model = PyTorchStereoCorrel(str(args.root/'calib.yaml'))
    rng = np.random.default_rng(20261005)
    rows = []
    for capture in sorted((args.root/'images').glob('*/laser')):
        name = capture.parent.name
        if args.captures and name not in args.captures:
            continue
        paths = [sorted((capture/cam).glob('*.png')) for cam in ['left', 'right']]
        if len(paths[0]) != len(paths[1]) or len(paths[0]) < 5:
            raise ValueError(f'{name}: expected at least five paired frames')
        if [p.stem[1:] for p in paths[0]] != [p.stem[1:] for p in paths[1]]:
            raise ValueError(f'{name}: frame identifiers do not match')
        images = [[cv2.imread(str(p), 0) for p in group[:5]] for group in paths]
        if any(img is None for group in images for img in group):
            raise ValueError(f'{name}: unreadable image')
        cloud = np.loadtxt(args.root/'output'/f'{name}-laser-5img-3kernel.csv', delimiter=',', skiprows=1, ndmin=2)
        # Uniform random sample of retained baseline sites plus broad XY probes.
        count = args.samples*3//4
        chosen = cloud[rng.choice(len(cloud), min(count, len(cloud)), replace=False)]
        broad = rng.integers(-199, 600, size=(args.samples-len(chosen), 2)).astype(np.float32)
        xy = torch.tensor(np.concatenate([chosen[:, :2], broad]), dtype=torch.float32, device=model.device)
        ref_z = np.concatenate([chosen[:, 2], np.full(len(broad), np.nan)])
        if args.sites_from:
            sites = np.load(args.sites_from/f'{name}_baseline.npz')
            take = np.linspace(0, len(sites['xy'])-1, args.samples, dtype=int)
            xy = torch.tensor(sites['xy'][take], dtype=torch.float32, device=model.device)
            ref_z = sites['reference_z'][take]
        fringe_file = args.root/'output'/f'{name}-fringe-1img-1kernel.csv'
        fringe_z = np.full(len(xy), np.nan)
        if fringe_file.exists():
            fringe = np.loadtxt(fringe_file, delimiter=',', skiprows=1, ndmin=2)
            distance, nearest = cKDTree(fringe[:, :2]).query(xy.cpu().numpy())
            matched = distance <= .75
            fringe_z[matched] = fringe[nearest[matched], 2]
        model.x_vals = torch.tensor([0., 1.], device=model.device)
        for variant, clahe, clip, blur, k in VARIANTS:
            if args.variants and variant not in args.variants:
                continue
            model.tuning_temporal = variant.startswith('temporal_')
            start = time.perf_counter()
            model.convert_images(*images, apply_clahe=clahe, undist=False,
                                 apply_blur=blur>1, blur=blur, climp=clip)
            # Independent full-range searches: all five, even frames, odd frames.
            dz_search = .1 if args.global_fine else 2.
            za, ca, gap = search(model, xy, k, list(range(5)), dz=dz_search)
            zt, _, _ = search(model, xy, k, [0, 2, 4], dz=dz_search)
            zv, _, _ = search(model, xy, k, [1, 3], dz=dz_search)
            if not args.global_fine:
                za, ca, _ = search(model, xy, k, list(range(5)), za, .1)
                zt, _, _ = search(model, xy, k, [0, 2, 4], zt, .1)
                zv, _, _ = search(model, xy, k, [1, 3], zv, .1)
            pts = torch.cat([xy, za[:, None]], 1)
            temporal_std = []
            for cam, imgs in [('left', model.left_images), ('right', model.right_images)]:
                uv, mask = model.transform_gcs2ccs(pts, cam, image_shape=imgs.shape[1:])
                temporal_std.append(model.interpolate_images(imgs, uv, mask).std(1))
            std = torch.minimum(*temporal_std)
            a, t, v, c, g, s = [x.cpu().numpy() for x in [za, zt, zv, ca, gap, std]]
            accepted = np.isfinite(c) & (c>.65) & (s>10)
            boundary = (a <= -799.9) | (a >= 799.9)
            disagreement = np.abs(t-v)
            safe = accepted & ~boundary & (g>.05)
            row = dict(capture=name, variant=variant, samples=len(xy), accepted=int(accepted.sum()),
                       boundary_accepted=int((accepted&boundary).sum()),
                       stable_1mm=int((accepted&~boundary&(disagreement<=1.)).sum()),
                       safe=int(safe.sum()), safe_stable_1mm=int((safe&(disagreement<=1.)).sum()),
                       split_median_mm=finite_median(disagreement[accepted&~boundary]),
                       baseline_median_mm=finite_median(np.abs(a-ref_z)[accepted]),
                       fringe_matched=int((accepted&np.isfinite(fringe_z)).sum()),
                       fringe_median_abs_mm=finite_median(np.abs(a-fringe_z)[accepted]),
                       fringe_within_1mm=int((accepted&(np.abs(a-fringe_z)<=1.)).sum()),
                       seconds=round(time.perf_counter()-start, 3))
            rows.append(row)
            np.savez_compressed(args.output/f'{name}_{variant}.npz', xy=xy.cpu().numpy(),
                                z=a, corr=c, gap=g, std=s, split_error=disagreement,
                                accepted=accepted, boundary=boundary, reference_z=ref_z,
                                fringe_z=fringe_z)
            print(json.dumps(row), flush=True)
            with (args.output/'summary.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    (args.output/'experiment.json').write_text(json.dumps(dict(root=str(args.root), samples=args.samples,
        seed=20261005, coarse_dz_mm=2., fine_dz_mm=.1, global_fine=args.global_fine, xy_patch_step_mm=1.,
        device=str(model.device), variants=[v for v in VARIANTS if v[0] in {r['variant'] for r in rows}], frames_train=[0,2,4], frames_validation=[1,3],
        limitations='Sampled sites, baseline-selected XY bias, frame-split proxy; no ground truth or GPU timing.'), indent=2))


if __name__ == '__main__':
    main()
