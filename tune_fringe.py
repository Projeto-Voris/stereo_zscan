"""Compare fringe interpolation on shared saved-cloud XY sites, full depth search.
Saved depths are regression references, not ground truth or search bounds.
"""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/stereo_zscan_mpl')
import cv2
import numpy as np
import torch
from include.SpatioCorrelation import PyTorchStereoCorrel
from include.fringe_projection.include.stereo_fringe_process import FringeProcess


@torch.inference_mode()
def search(model, xy, dz):
    levels = torch.arange(-800., 800.+dz/2, dz, device=model.device)
    best = torch.full((len(xy),), torch.inf, device=model.device)
    depth = torch.zeros_like(best)
    for chunk in levels.split(64):
        points = torch.cat([xy[None].expand(len(chunk), -1, -1),
                            chunk[:, None, None].expand(-1, len(xy), 1)], 2).reshape(-1, 3)
        values = []
        for cam, images in [('left', model.left_images), ('right', model.right_images)]:
            uv, valid = model.transform_gcs2ccs(points, cam, image_shape=images.shape[1:])
            method = 'correl' if model.variant == 'original' else 'fringe'
            values.append(model.interpolate_images(images, uv, valid, method=method))
        scores = (values[0][:, 0]-values[1][:, 0]).abs()
        mod_valid = (values[0][:, 1] > 10) & (values[1][:, 1] > 10)
        # Original production selects a minimum before the modulation mask.
        if model.variant == 'guarded':
            scores = scores.masked_fill(~mod_valid, torch.inf)
        scores = torch.nan_to_num(scores, nan=torch.inf).reshape(len(chunk), -1)
        score, index = scores.min(0)
        update = score < best
        depth[update], best[update] = chunk[index[update]], score[update]
    pts = torch.cat([xy, depth[:, None]], 1)
    valid = (best < .1) & (depth > -800) & (depth < 800)
    for cam, images in [('left', model.left_images), ('right', model.right_images)]:
        uv, mask = model.transform_gcs2ccs(pts, cam, image_shape=images.shape[1:])
        method = 'correl' if model.variant == 'original' else 'fringe'
        samples = model.interpolate_images(images, uv, mask, method=method)
        valid &= samples[:, 1] > 10
    return depth.cpu().numpy(), best.cpu().numpy(), valid.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/home/daniel/Pictures/20260928 - Avaliação reconstrução sistemas'))
    parser.add_argument('--output', type=Path, default=Path('fringe_tuning'))
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--dz', type=float, default=.1)
    parser.add_argument('--captures', nargs='*')
    args = parser.parse_args()
    if args.samples < 1 or args.dz <= 0:
        parser.error('samples and dz must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    model = PyTorchStereoCorrel(str(args.root/'calib.yaml'))
    rng = np.random.default_rng(20261005)
    rows = []
    for capture in sorted(args.root.glob('*/fringe')):
        name = capture.parent.name
        if args.captures and name not in args.captures:
            continue
        clouds = list(capture.glob('*-fringe-1img-1kernel.csv'))
        if not clouds:
            continue
        cloud = np.loadtxt(clouds[0], delimiter=',', skiprows=1, ndmin=2)
        chosen = cloud[rng.choice(len(cloud), min(args.samples, len(cloud)), replace=False)]
        xy = torch.as_tensor(chosen[:, :2], dtype=torch.float32, device=model.device)
        paths = [sorted((capture/cam).glob('*.png')) for cam in ['left', 'right']]
        if len(paths[0]) != 16 or len(paths[1]) != 16 or [p.stem[1:] for p in paths[0]] != [p.stem[1:] for p in paths[1]]:
            raise ValueError(f'{name}: expected 16 matching Gray-code + phase frames')
        images = [[cv2.imread(str(p), 0) for p in group] for group in paths]
        if any(im is None for group in images for im in group):
            raise ValueError(f'{name}: unreadable image')
        processor = FringeProcess(camera_resolution=images[0][0].shape[::-1], px_f=64, steps=8)
        processor.images_left, processor.images_right = [np.stack(group, -1) for group in images]
        al, ar, ml, mr = processor.calculate_abs_phi_images(visualize=False)
        model.convert_images([al.cpu().numpy(), ml.cpu().numpy()], [ar.cpu().numpy(), mr.cpu().numpy()], apply_clahe=False, undist=False)
        del images
        results = {}
        for variant in ['original', 'bilinear', 'guarded']:
            model.variant = variant
            model.fringe_max_phase_jump = np.pi if variant == 'guarded' else np.inf
            model.fringe_min_modulation = 10 if variant == 'guarded' else 0
            z, score, valid = search(model, xy, args.dz)
            results[variant] = (z, score, valid)
            error = abs(z-chosen[:, 2])
            row = dict(capture=name, variant=variant, sampled=len(xy), retained=int(valid.sum()),
                       reference_within_1mm=int((valid & (error <= 1)).sum()),
                       reference_median_mm=float(np.median(error[valid])) if valid.any() else None)
            rows.append(row)
            print(row, flush=True)
        # Independent sinusoidal subsets share Gray code and calibration.
        # Keep the original phase angles, avoiding a phase-origin shift.
        for split, indices in [('even', [0, 2, 4, 6]), ('odd', [1, 3, 5, 7])]:
            def calculate_subset(image, name='Plot', visualize=False, indices=indices):
                image = processor._to_tensor(image, dtype=torch.float32)
                angles = 2*np.pi*(torch.tensor(indices, device=image.device)+1)/8
                selected = image[:, :, indices]
                sine = (selected*angles.sin()).sum(2)
                cosine = (selected*angles.cos()).sum(2)
                return (sine.square()+cosine.square()).sqrt()/len(indices), torch.atan2(-sine, cosine)
            processor.calculate_phi = calculate_subset
            al, ar, ml, mr = processor.calculate_abs_phi_images(visualize=False)
            model.convert_images([al.cpu().numpy(), ml.cpu().numpy()], [ar.cpu().numpy(), mr.cpu().numpy()], apply_clahe=False, undist=False)
            for variant in ['original', 'guarded']:
                model.variant = variant
                model.fringe_max_phase_jump = np.pi if variant == 'guarded' else np.inf
                model.fringe_min_modulation = 10 if variant == 'guarded' else 0
                results[f'{variant}_{split}'] = search(model, xy, args.dz)
        for variant in ['original', 'guarded']:
            za, _, va = results[f'{variant}_even']
            zb, _, vb = results[f'{variant}_odd']
            z, _, valid = results[variant]
            stable = valid & va & vb & (abs(za-zb) <= 1.)
            row = next(row for row in rows if row['capture'] == name and row['variant'] == variant)
            row['split_both_retained'] = int((valid & va & vb).sum())
            row['split_within_1mm'] = int(stable.sum())
            print('split', row, flush=True)
        del processor
        np.savez_compressed(args.output/f'{name}.npz' , xy=chosen[:, :2], reference_z=chosen[:, 2],
                            **{f'{v}_{field}': data[i] for v, data in results.items() for i, field in enumerate(['z', 'score', 'valid'])})
        (args.output/'summary.json').write_text(json.dumps(rows, indent=2)+'\n')
    (args.output/'settings.json').write_text(json.dumps(dict(root=str(args.root), samples=args.samples, dz=args.dz, range_mm=[-800, 800]), indent=2)+'\n')


if __name__ == '__main__':
    main()
