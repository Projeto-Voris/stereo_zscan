import sys
import os
import argparse
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

import numpy as np
import torch
import time
import cv2
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
from include.fringe_projection.include.stereo_fringe_process import FringeProcess
from include.SpatioCorrelation import PyTorchStereoCorrel
from include.laser_surface import supported_components, depth_bands, select_depths, refine_regions
# from include.SpatioTemporal_triton import SpatioTemporalTriton as PyTorchStereoCorrel

def save_point_cloud(filename: str | Path, xyz: torch.Tensor, corr: Optional[torch.Tensor] = None, delimiter: str = ','):
    """Salva uma nuvem de pontos XYZ, opcionalmente com valores de correlação."""
    if isinstance(xyz, torch.Tensor):
        xyz = xyz.cpu().numpy()
    if corr is not None and isinstance(corr, torch.Tensor):
        corr = corr.cpu().numpy()

    if corr is not None:
        if corr.ndim == 1:
            corr = corr[:, None]
        data = np.hstack((xyz, corr))
        header_str = 'x,y,z,corr'
    else:
        data = xyz
        header_str = 'x,y,z'
        
    np.savetxt(filename, data, delimiter=delimiter, header=header_str, comments='')
    print(f"Nuvem de pontos salva em {filename}")


def parse_args():
    parser = argparse.ArgumentParser(description='Dense inverse-triangulation reconstruction')
    parser.add_argument('--calib', type=Path, default=Path('/home/daniel/Pictures/20261001 - Avaliação reconstrução sistemas ar/active_stereo/calib.yaml'))
    parser.add_argument('--image-root', type=Path, default=Path('/home/daniel/Pictures/20261001 - Avaliação reconstrução sistemas ar/active_stereo/images'))
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'reconstruction_fringe')
    parser.add_argument('--objects', nargs='+', help='Capture folder names; defaults to all captures')
    parser.add_argument('--x-limits', nargs=2, type=float, default=(-200., 600.), metavar=('MIN', 'MAX'))
    parser.add_argument('--y-limits', nargs=2, type=float, default=(-200., 600.), metavar=('MIN', 'MAX'))
    parser.add_argument('--z-limits', nargs=2, type=float, default=(-800., 800.), metavar=('MIN', 'MAX'))
    parser.add_argument('--laser-preset', choices=['baseline', 'balanced', 'precision'], default='balanced',
                        help='baseline: original 3x3; balanced: 5x5 + blur3, corr>.70; precision: 5x5 + blur3, corr>.80')
    parser.add_argument('--methods', nargs='+', choices=['laser', 'fringe', 'correl'], default=['fringe'])
    parser.add_argument('--fringe-modulation', type=float, default=10., help='Minimum fringe modulation at every interpolation support pixel')
    parser.add_argument('--no-plot', action='store_true', help='Run reconstruction without interactive plots')
    parser.add_argument('--neighbor-backend', choices=['torch', 'kdtree'], default='torch',
                        help='kdtree avoids the quadratic all-pairs neighbor matrix')
    parser.add_argument('--laser-support', choices=['connected', 'off'], default='connected',
                        help='Reject small 3D components and refine only supported depth bands')
    parser.add_argument('--surface-radius', type=float, default=5., help='3D connectivity radius in mm')
    parser.add_argument('--surface-min-points', type=int, default=16, help='Minimum supported component size')
    parser.add_argument('--depth-band-margin', type=float, default=10., help='Padding around coarse component depth bands in mm')
    parser.add_argument('--laser-split-tolerance', type=float, default=1.,
                        help='Maximum fine-pass frame-split depth difference in mm; negative disables validation')
    parser.add_argument('--surface-seeds', type=Path,
                        help='Reuse existing clouds of these same captures as geometry seeds, skipping the coarse scan')
    parser.add_argument('--refine-tile-size', type=float, default=32.,
                        help='Fine refinement neighborhoods in mm; zero uses whole component boxes')
    return parser.parse_args()


def main(args=None):
    """Função principal para executar o pipeline de correlação estéreo."""
    
    args = parse_args() if args is None else args
    YAML_FILE = str(args.calib)

    distances =[0]
    offset = 0
    dz = 800
    plot = not args.no_plot #plot 3D points of two iterations
    debug = False #plot debug images
    save_file = True # save csv file with point cloud
    euclidian_filter = True # apply euclidean filter
    mask_std = True # apply STD MASK\n
    mask_uv = True # apply uv mask
    methods = args.methods

    GRID_STEPS_1 = {'xy': 1.0, 'z': 2.0} # first steps of 3d patch
    GRID_STEPS_2 = {'xy': 1.0, 'z': 0.1} # second steps of 3d patch
    
    Zscan = PyTorchStereoCorrel(yaml_file=YAML_FILE)
    neighbor_filter = Zscan.euclidean_filter if args.neighbor_backend == 'torch' else Zscan.euclidean_filter_kdtree
    Zscan.fringe_min_modulation = args.fringe_modulation
    PATH = str(args.image_root)
    objects = args.objects or sorted(p.name for p in args.image_root.iterdir() if p.is_dir())
    black_factor = 1# 0 to 1 

    for obj in objects:
        for method in methods:
            for dist in distances:
                print('=' * 50)
                print(f"\nIniciando processamento para {obj} usando o método {method}...")
                # IMAGES_PATH = Path('20250513_1505_step10_calota_d2')
                # IMAGES_PATH = Path('./correl/{}/{}'.format( obj, dist))
                # IMAGES_PATH = Path('./testes/{}/{}_{}_{}'.format(method, method, obj, dist))
                IMAGES_PATH = Path(f'{PATH}/{obj}/{method}')
                if not IMAGES_PATH.exists() or not any(IMAGES_PATH.iterdir()):
                    print(f"Skipping '{IMAGES_PATH}': Folder does not exist or is empty.")
                    continue
                    
                if method == 'laser':
                    N_IMGS_OPTIONS = [5]
                    laser_presets = {'baseline': (3, 1, .65),
                                     'balanced': (5, 3, .70),
                                     'precision': (5, 3, .80)}
                    laser_kernel, laser_blur, FILTER_THRESHOLD = laser_presets[args.laser_preset]
                    KERNEL_SIZES = [laser_kernel]
                    print(f'Laser preset: {args.laser_preset}, world patch {laser_kernel}x{laser_kernel}, image blur {laser_blur}x{laser_blur}')
                    FILTER_RADIUS = 10.0
                    FILTER_MIN_NEIGHBORS = 5
                    MASK_BOUNDS = 10 # Limites do desvio padrão dos pontos interpolados nas T imagens
                use_surface_support = method == 'laser' and args.laser_support == 'connected'
                validate_split = use_surface_support and args.laser_split_tolerance >= 0
                laser_stats = {}
                if method == 'correl':
                    N_IMGS_OPTIONS = [15]
                    KERNEL_SIZES = [1]
                    FILTER_THRESHOLD = 0.9
                    FILTER_RADIUS = 15.0
                    FILTER_MIN_NEIGHBORS = 5
                    MASK_BOUNDS = 20 # Limites do desvio padrão dos pontos interpolados nas T imagens

                if method == 'fringe':
                    N_IMGS_OPTIONS = [1]
                    KERNEL_SIZES = [1]
                    PIXEL_PER_FRINGE = 64
                    STEPS = 8
                    FILTER_THRESHOLD = 0.1
                    FILTER_RADIUS = 10.0
                    FILTER_MIN_NEIGHBORS = 10
                    MASK_BOUNDS = args.fringe_modulation # Limites da modulação dos pontos interpolados
                
                
                GRID_LIMITS = {'x': tuple(args.x_limits), 'y': tuple(args.y_limits),
                               'z': tuple(args.z_limits)}

                current_timestamp = time.strftime("%Y%m%d")
                out = Path('{}-{}'.format(current_timestamp, method))
                output_path = args.output
                output_path.mkdir(parents=True, exist_ok=True)

                t_start_total = time.time()
                
                try:
                    left_path = IMAGES_PATH / 'left'
                    right_path = IMAGES_PATH / 'right'
                    left_imgs_list = sorted([p.name for p in left_path.iterdir()])
                    right_imgs_list = sorted([p.name for p in right_path.iterdir()])
                    if not left_imgs_list or not right_imgs_list:
                        print(f"Erro: Não foram encontradas imagens em {IMAGES_PATH}")
                        return
                except FileNotFoundError:
                    print(f"Erro: Diretório de imagens não encontrado: {IMAGES_PATH}")
                    return
                    
                print('Imagens encontradas. Processamento iniciado...')

                def read_images_from_disk(path: Path, images_list: list, n_imgs: int) -> list:
                    return [cv2.imread(str(path / img_name), cv2.IMREAD_GRAYSCALE) for img_name in images_list[:n_imgs]]
                

                




                if method == 'fringe':
                    left_imgs_cpu = read_images_from_disk(left_path, left_imgs_list, n_imgs=len(left_imgs_list))
                    left_imgs_cpu = [cv2.convertScaleAbs(img, alpha=black_factor) for img in left_imgs_cpu]
                    right_imgs_cpu = read_images_from_disk(right_path, right_imgs_list, n_imgs=len(right_imgs_list))
                    right_imgs_cpu = [cv2.convertScaleAbs(img, alpha=black_factor) for img in right_imgs_cpu]
                    fringe_imgs_proc = FringeProcess(camera_resolution=left_imgs_cpu[0].shape[::-1], px_f=PIXEL_PER_FRINGE, steps=STEPS)
                    fringe_imgs_proc.images_left = np.moveaxis(np.array(left_imgs_cpu), 0, -1)
                    fringe_imgs_proc.images_right = np.moveaxis(np.array(right_imgs_cpu),0, -1)
                    abs_left, abs_right, mod_left, mod_right = fringe_imgs_proc.calculate_abs_phi_images(visualize=debug)
                    left_imgs_cpu = [abs_left.cpu().numpy(), mod_left.cpu().numpy()]
                    right_imgs_cpu = [abs_right.cpu().numpy(), mod_right.cpu().numpy()]

                for n_img in N_IMGS_OPTIONS:

                    if method != 'fringe':
                        print(f'Carregando {n_img} pares de imagens...')
                        left_imgs_cpu = read_images_from_disk(left_path, left_imgs_list, n_imgs=n_img)
                        left_imgs_cpu = [cv2.convertScaleAbs(img, alpha=black_factor) for img in left_imgs_cpu]
                        right_imgs_cpu = read_images_from_disk(right_path, right_imgs_list, n_imgs=n_img)
                        right_imgs_cpu = [cv2.convertScaleAbs(img, alpha=black_factor) for img in right_imgs_cpu]


                    for kernel in KERNEL_SIZES:
                        print(f"Iniciando {method} com kernel espacial {kernel}x{kernel}")
                        
                        t_run_start = time.time()
                        preset_suffix = f'-{args.laser_preset}' if method == 'laser' and args.laser_preset != 'baseline' else ''
                        if use_surface_support:
                            preset_suffix += '-supported'
                        cloud_file = output_path / f'{obj}-{method}-{n_img}img-{kernel}kernel{preset_suffix}.csv'
                                        
                        print("Convertendo imagens (CLAHE, Undistort)...")
                        if method == 'fringe':
                            Zscan.convert_images(left_imgs_cpu, right_imgs_cpu, apply_clahe=False, undist=False)
                        else:
                            blur_size = laser_blur if method == 'laser' else 1
                            Zscan.convert_images(left_imgs_cpu, right_imgs_cpu, apply_clahe=True, undist=False,
                                                 apply_blur=blur_size > 1, blur=blur_size)
                        del left_imgs_cpu, right_imgs_cpu
                            

                        t_preprocessing_done = time.time()
                        print(f"Pré-processamento de imagens concluído em {t_preprocessing_done - t_run_start:.2f} s")

                        print("Construindo grade 3D e iniciando a correlação...")
                        Zscan.points3d(x_lim=GRID_LIMITS['x'], y_lim=GRID_LIMITS['y'], z_lim=GRID_LIMITS['z'],
                                    xy_step=GRID_STEPS_1['xy'], z_step=GRID_STEPS_1['z'])
                        
                        if method == 'laser' and args.surface_seeds:
                            if not use_surface_support:
                                raise ValueError('--surface-seeds requires connected laser support')
                            seed_files = sorted(args.surface_seeds.glob(f'{obj}-laser*.csv'))
                            if len(seed_files) != 1:
                                raise ValueError(f'Expected one laser seed cloud for {obj}, found {len(seed_files)}')
                            seeds = np.loadtxt(seed_files[0], delimiter=',', skiprows=1, ndmin=2)
                            xyz_gpu = torch.as_tensor(seeds[:,:3], dtype=torch.float32, device=Zscan.device)
                            corr_gpu = torch.as_tensor(seeds[:,3], dtype=torch.float32, device=Zscan.device)
                            laser_stats['seed_file'] = str(seed_files[0].resolve())
                            print(f'Reusing {len(xyz_gpu)} geometry seeds from {seed_files[0]}; coarse search skipped')
                        else:
                            xyz_gpu, corr_gpu, _ = Zscan.process_segmented_z(Kx=kernel, Ky=kernel, stride=1, Nz_block_voxels=10, method=method)

                        t_correlation_done = time.time()
                        laser_stats['coarse_seconds'] = t_correlation_done - t_preprocessing_done
                        print(f"1a Correlação concluída em {t_correlation_done - t_preprocessing_done:.2f} s")

                        if xyz_gpu.numel() == 0:
                            print(f"Nenhum ponto retornado pelo processamento para.")
                            continue

                        print(f"Total de pontos brutos: {xyz_gpu.shape[0]}")
                        if method == 'fringe':
                            filter_mask = corr_gpu < FILTER_THRESHOLD
                            xyz_filtered_gpu = xyz_gpu[filter_mask]
                            corr_filtered_gpu = corr_gpu[filter_mask]
                            print(f"Pontos com correlação < {FILTER_THRESHOLD} rad: {xyz_filtered_gpu.shape[0]}")
                        else:
                            # Seed scores came from a previous patch/preprocessing
                            # configuration. Use seed geometry, not those old scores,
                            # to restrict the new fine-pass hypotheses.
                            if method == 'laser' and args.surface_seeds:
                                filter_mask = torch.isfinite(corr_gpu) & torch.isfinite(xyz_gpu).all(1)
                            else:
                                filter_mask = (corr_gpu > FILTER_THRESHOLD) & torch.isfinite(corr_gpu)
                            if method == 'laser' and (args.laser_preset != 'baseline' or use_surface_support):
                                filter_mask &= (xyz_gpu[:, 2] > Zscan.z_vals[0]) & (xyz_gpu[:, 2] < Zscan.z_vals[-1])
                            xyz_filtered_gpu = xyz_gpu[filter_mask]
                            corr_filtered_gpu = corr_gpu[filter_mask]
                            label = 'Finite geometry seeds' if method == 'laser' and args.surface_seeds else f'Pontos com correlação > {FILTER_THRESHOLD*100}%'
                            print(f'{label}: {xyz_filtered_gpu.shape[0]}')
                        
                        if debug:
                            Zscan.plot_3d_points(x=xyz_filtered_gpu[:,0],
                                                y=xyz_filtered_gpu[:,1],
                                                z=xyz_filtered_gpu[:,2],
                                                color=corr_filtered_gpu,
                                                title=f'1a {method} - Pontos 3D: {dist} mm {n_img} imgs {kernel}x{kernel}')

                        interp_filtered = None
                        if mask_uv:
                            xyz_filtered_gpu, corr_filtered_gpu, interp_filtered = Zscan.mask_uv_points(xyz_filtered_gpu, corr_filtered_gpu)
                            if debug:
                                Zscan.plot_3d_points(x=xyz_filtered_gpu[:,0],
                                                    y=xyz_filtered_gpu[:,1],
                                                    z=xyz_filtered_gpu[:,2],
                                                    color=interp_filtered,
                                                    title=f'DEBUG UV MASK {method} - Pontos 3D filtrados UV: {dist} mm {n_img} imgs {kernel}x{kernel}')
                        if mask_std:
                            xyz_filtered_gpu, corr_filtered_gpu, interp_filtered = Zscan.std_mask_points(xyz_filtered_gpu, corr_filtered_gpu, bounds=MASK_BOUNDS, method=method)
                            if debug:
                                Zscan.plot_3d_points(x=xyz_filtered_gpu[:,0],
                                                    y=xyz_filtered_gpu[:,1],
                                                    z=xyz_filtered_gpu[:,2],
                                                    color=interp_filtered,
                                                    title=f'DEBUG STD MASK\n {method} - Pontos 3D filtrados UV: {dist} mm {n_img} imgs {kernel}x{kernel}')

                        if euclidian_filter and not use_surface_support:
                            xyz_filtered_gpu, corr_filtered_gpu, interp = neighbor_filter(xyz_gpu=xyz_filtered_gpu, corr_gpu=corr_filtered_gpu, interp=interp_filtered,
                                                                        min_neighbors=FILTER_MIN_NEIGHBORS, radius=FILTER_RADIUS)

                        if use_surface_support:
                            coarse_before = len(xyz_filtered_gpu)
                            raw_z = xyz_filtered_gpu[:, 2].cpu().numpy()
                            if len(raw_z):
                                laser_stats['unpruned_fine_depth_candidates'] = int(np.ceil((raw_z.max()-raw_z.min()+4.)/.1))+1
                            support_mask, components = supported_components(xyz_filtered_gpu.cpu().numpy(),
                                radius=args.surface_radius, min_points=args.surface_min_points)
                            support_mask = torch.as_tensor(support_mask, device=Zscan.device)
                            xyz_filtered_gpu, corr_filtered_gpu = xyz_filtered_gpu[support_mask], corr_filtered_gpu[support_mask]
                            bands = depth_bands(components, margin=args.depth_band_margin, limits=GRID_LIMITS['z'])
                            laser_stats.update(coarse_before_support=coarse_before, coarse_supported=len(xyz_filtered_gpu),
                                               components=components, depth_bands=bands)
                            print(f'Coarse surface support: {coarse_before} -> {len(xyz_filtered_gpu)} points, {len(components)} components')
                                                            
                        if xyz_filtered_gpu.numel() == 0:
                            print('No points survived coarse filtering; skipping refinement.')
                            if use_surface_support:
                                laser_stats.update(final_points=0, fine_seconds=0., total_seconds=time.time()-t_run_start)
                                (output_path / f'{obj}-laser-diagnostics.json').write_text(json.dumps(laser_stats, indent=2))
                                save_point_cloud(cloud_file, xyz_filtered_gpu, corr_filtered_gpu)
                            continue

                        # Find first 3D bounds to refined process
                        xlim = torch.min(xyz_filtered_gpu[:, 0]), torch.max(xyz_filtered_gpu[:, 0])
                        ylim = torch.min(xyz_filtered_gpu[:, 1]), torch.max(xyz_filtered_gpu[:, 1])
                        zlim = torch.min(xyz_filtered_gpu[:, 2]), torch.max(xyz_filtered_gpu[:, 2])

                        # self.get_logger().info(f'3D bounds for refined triangulation: X: {xlim}, Y: {ylim}, Z: {zlim}')
                        zlo, zhi = float(zlim[0]), float(zlim[1])
                        if method == 'fringe' or (method == 'laser' and args.laser_preset != 'baseline') or zlo == zhi:
                            margin = GRID_STEPS_1['z']
                            zlo = max(GRID_LIMITS['z'][0], zlo - margin)
                            zhi = min(GRID_LIMITS['z'][1], zhi + margin)
                        zlim = (zlo, zhi)

                        if use_surface_support:
                            # Preserve coarse edge centers when a new fine patch is built.
                            pad = (kernel // 2 + 6) * GRID_STEPS_2['xy']
                            xlim = (max(GRID_LIMITS['x'][0], float(xlim[0])-pad), min(GRID_LIMITS['x'][1], float(xlim[1])+pad))
                            ylim = (max(GRID_LIMITS['y'][0], float(ylim[0])-pad), min(GRID_LIMITS['y'][1], float(ylim[1])+pad))
                            zlim = (bands[0][0], bands[-1][1])

                        Zscan.points3d(x_lim=xlim, y_lim=ylim, z_lim=zlim, xy_step=GRID_STEPS_2['xy'], z_step=GRID_STEPS_2['z'])

                        if use_surface_support:
                            depth_mask = select_depths(Zscan.z_vals.cpu().numpy(), bands)
                            Zscan.z_vals = Zscan.z_vals[torch.as_tensor(depth_mask, device=Zscan.device)]
                            laser_stats['fine_depth_candidates'] = len(Zscan.z_vals)
                            laser_stats['fine_xy_grid'] = [len(Zscan.x_vals), len(Zscan.y_vals)]
                            print(f"Fine depth search: {len(Zscan.z_vals)} candidates in {len(bands)} supported bands "
                                  f"(unpruned: {laser_stats.get('unpruned_fine_depth_candidates', 0)})")

                        fine_start = time.time()
                        if use_surface_support:
                            xyz_gpu, corr_gpu, region_stats = refine_regions(Zscan, components, kernel,
                                z_margin=args.depth_band_margin, validate_split=validate_split,
                                seed_points=xyz_filtered_gpu.cpu().numpy(), tile_size=args.refine_tile_size)
                            laser_stats.update(region_stats)
                            print(f"Component refinement: {region_stats['fine_projected_voxels']} projected voxels "
                                  f"instead of {region_stats['rectangular_band_voxels']} in the enclosing banded box")
                        else:
                            xyz_gpu, corr_gpu, _ = Zscan.process_segmented_z(Kx=kernel, Ky=kernel, stride=1,
                                                                            Nz_block_voxels=10, method=method)
                        laser_stats['fine_seconds'] = time.time() - fine_start


                        if xyz_gpu.numel() == 0:
                            print(f"Nenhum ponto retornado pelo processamento para.")
                            if use_surface_support:
                                laser_stats.update(final_points=0, total_seconds=time.time()-t_run_start)
                                (output_path / f'{obj}-laser-diagnostics.json').write_text(json.dumps(laser_stats, indent=2))
                                save_point_cloud(cloud_file, xyz_gpu, corr_gpu)
                            continue

                        print(f"Total de pontos brutos: {xyz_gpu.shape[0]}")
                        if method == 'fringe':
                            filter_mask = corr_gpu < FILTER_THRESHOLD
                            xyz_filtered_gpu = xyz_gpu[filter_mask]
                            corr_filtered_gpu = corr_gpu[filter_mask]
                            print(f"Pontos com correlação < {FILTER_THRESHOLD} rad: {xyz_filtered_gpu.shape[0]}")
                        else:
                            filter_mask = (corr_gpu > FILTER_THRESHOLD) & torch.isfinite(corr_gpu)
                            if method == 'laser' and (args.laser_preset != 'baseline' or use_surface_support):
                                filter_mask &= (xyz_gpu[:, 2] > Zscan.z_vals[0]) & (xyz_gpu[:, 2] < Zscan.z_vals[-1])
                            if validate_split:
                                before_split = int(filter_mask.sum())
                                filter_mask &= Zscan.laser_diagnostics['split_depth_error'] <= args.laser_split_tolerance
                                laser_stats.update(fine_before_split=before_split, fine_after_split=int(filter_mask.sum()))
                                print(f'Fine frame-split consistency: {before_split} -> {int(filter_mask.sum())}')
                            xyz_filtered_gpu = xyz_gpu[filter_mask]
                            corr_filtered_gpu = corr_gpu[filter_mask]
                            print(f"Pontos com correlação > {FILTER_THRESHOLD*100}%: {xyz_filtered_gpu.shape[0]}")
                        
                        if debug:
                            Zscan.plot_3d_points(x=xyz_filtered_gpu[:,0],
                                                y=xyz_filtered_gpu[:,1],
                                                z=xyz_filtered_gpu[:,2],
                                                color=corr_filtered_gpu,
                                                title=f'1a {method} - Pontos 3D: {dist} mm {n_img} imgs {kernel}x{kernel}')

                        interp_filtered = None
                        if mask_uv:
                            xyz_filtered_gpu, corr_filtered_gpu, interp_filtered = Zscan.mask_uv_points(xyz_filtered_gpu, corr_filtered_gpu)
                            if debug:
                                Zscan.plot_3d_points(x=xyz_filtered_gpu[:,0],
                                                    y=xyz_filtered_gpu[:,1],
                                                    z=xyz_filtered_gpu[:,2],
                                                    color=interp_filtered,
                                                    title=f'DEBUG UV MASK {method} - Pontos 3D filtrados UV: {dist} mm {n_img} imgs {kernel}x{kernel}')
                        if mask_std:
                            xyz_filtered_gpu, corr_filtered_gpu, interp_filtered = Zscan.std_mask_points(xyz_filtered_gpu, corr_filtered_gpu, bounds=MASK_BOUNDS, method=method)
                            if debug:
                                Zscan.plot_3d_points(x=xyz_filtered_gpu[:,0],
                                                    y=xyz_filtered_gpu[:,1],
                                                    z=xyz_filtered_gpu[:,2],
                                                    color=interp_filtered,
                                                    title=f'DEBUG STD MASK\n {method} - Pontos 3D filtrados UV: {dist} mm {n_img} imgs {kernel}x{kernel}')

                        if use_surface_support:
                            support_mask, final_components = supported_components(xyz_filtered_gpu.cpu().numpy(),
                                radius=args.surface_radius, min_points=args.surface_min_points)
                            support_mask = torch.as_tensor(support_mask, device=Zscan.device)
                            final_xyz_gpu, final_corr_gpu = xyz_filtered_gpu[support_mask], corr_filtered_gpu[support_mask]
                            laser_stats.update(final_before_support=len(xyz_filtered_gpu), final_points=len(final_xyz_gpu),
                                               final_components=final_components)
                        elif euclidian_filter:
                            final_xyz_gpu, final_corr_gpu, interp = neighbor_filter(xyz_gpu=xyz_filtered_gpu, corr_gpu=corr_filtered_gpu, interp=interp_filtered,
                                                                        min_neighbors=FILTER_MIN_NEIGHBORS, radius=FILTER_RADIUS)
                        else:
                            final_xyz_gpu = xyz_filtered_gpu
                            final_corr_gpu = corr_filtered_gpu

                        # Construct second 3d points
                        

                        if final_xyz_gpu.numel() > 0 or use_surface_support:
                            if save_file:
                                save_point_cloud(cloud_file, final_xyz_gpu, final_corr_gpu)

                        t_run_end = time.time()
                        if use_surface_support:
                            laser_stats.update(total_seconds=t_run_end-t_run_start, preset=args.laser_preset,
                                               split_tolerance=args.laser_split_tolerance,
                                               surface_radius=args.surface_radius, surface_min_points=args.surface_min_points,
                                               refine_tile_size=args.refine_tile_size)
                            (output_path / f'{obj}-laser-diagnostics.json').write_text(json.dumps(laser_stats, indent=2))
                        print(f"======== Concluído: em {t_run_end - t_run_start:.2f} s ========")
                        if plot:
                            Zscan.plot_3d_points(x=final_xyz_gpu[:,0],
                                                y=final_xyz_gpu[:,1],
                                                z=final_xyz_gpu[:,2],
                                                color=final_corr_gpu,
                                                title=f'2a  {method} - Pontos 3D: {dist} mm {n_img} imgs {kernel}x{kernel}')

                t_end_total = time.time()
                print(f"\nProcessamento total concluído em {t_end_total - t_start_total:.2f} s.")
                torch.cuda.empty_cache()
                print(f"Resultados salvos em: {output_path.resolve()}")

if __name__ == "__main__":
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU detectada: {props.name}, Memória Total: {props.total_memory / (1024**2):.2f} MB")
    else:
        print("GPU não detectada pelo PyTorch. O código será executado na CPU, o que pode ser muito lento.")
    main()
