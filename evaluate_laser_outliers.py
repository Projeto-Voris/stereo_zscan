"""Inspect saved laser clouds and write separate geometric postfilter results."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/stereo_zscan_mpl')
import numpy as np
from scipy.spatial import cKDTree
from include.laser_surface import supported_components, depth_bands


def reference_metrics(cloud, mask, reference):
    ref = np.loadtxt(reference, delimiter=',', skiprows=1, ndmin=2)
    distance, index = cKDTree(ref[:, :2]).query(cloud[:, :2])
    overlap = distance <= .75
    agree = overlap & (abs(cloud[:, 2]-ref[index, 2]) <= 1.)
    return dict(overlap_before=int(overlap.sum()), overlap_after=int((overlap&mask).sum()),
                agree_before=int(agree.sum()), agree_after=int((agree&mask).sum()),
                agreeing_point_retention=float((agree&mask).sum()/max(agree.sum(), 1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('reconstruction'))
    parser.add_argument('--output', type=Path, default=Path('reconstruction_filtered'))
    parser.add_argument('--radius', type=float, default=5.)
    parser.add_argument('--min-points', type=int, default=32)
    parser.add_argument('--reference-root', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted(args.input.glob('*laser*.csv')):
        cloud = np.loadtxt(path, delimiter=',', skiprows=1, ndmin=2)
        valid = np.isfinite(cloud).all(1) & (abs(cloud[:, 2]) < 799.9)
        local, components = supported_components(cloud[valid, :3], args.radius, args.min_points)
        mask = np.zeros(len(cloud), bool)
        mask[valid] = local
        bands = depth_bands(components)
        filtered = cloud[mask]
        np.savetxt(args.output/(path.stem+'-supported-postfilter.csv'), filtered,
                   delimiter=',', header='x,y,z,corr', comments='')
        row = dict(file=path.name, before=len(cloud), after=len(filtered), rejected=int((~mask).sum()),
                   original_z_range=[float(cloud[:,2].min()),float(cloud[:,2].max())],
                   supported_z_range=[float(filtered[:,2].min()),float(filtered[:,2].max())] if len(filtered) else None,
                   bands=bands, band_width_mm=float(sum(hi-lo for lo,hi in bands)), components=components)
        if args.reference_root:
            ref = args.reference_root/path.name.replace('laser-5img-3kernel', 'fringe-1img-1kernel')
            if ref.exists():
                row['fringe_comparison'] = reference_metrics(cloud, mask, ref)
        records.append(row)
        print(json.dumps({k:v for k,v in row.items() if k!='components'}), flush=True)
    (args.output/'surface_filter_summary.json').write_text(json.dumps(dict(radius_mm=args.radius,
        min_component_points=args.min_points, filter='geometric postfilter only; no frame validation or new triangulation',
        captures=records), indent=2))


if __name__ == '__main__':
    main()
