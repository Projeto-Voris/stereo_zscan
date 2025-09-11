import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
import glob
import numpy as np
import torch
import time
import cv2
from pathlib import Path
from typing import Optional
from include.fringe_projection.include.stereo_fringe_process import FringeProcess
from include.SpatialCorrelation_pytorch import PyTorchStereoCorrel


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


def main():
    """Função principal para executar o pipeline de correlação estéreo."""
    
    YAML_FILE = 'cfg/SM4.yaml' # Yaml file with camera parameters
    # YAML_FILE = 'cfg/SM4.yaml' # Yaml file with camera parameters
    PATH = Path('g:\Drives compartilhados\VORIS - Media 2\\20250828 - Active stereo evaluation\\testes')
    dz = 350
    plot = True #plot 3D points of two iterations
    debug = True #plot debug images
    euclidian_filter = True # apply euclidean filter
    mask_std = True # apply STD MASK\n
    mask_uv = True # apply uv mask
    method = ['laser', 'fringe','pattern']
    files = {mtd: sorted(os.listdir(Path(PATH / mtd))) for mtd in method}
    GRID_STEPS_1 = {'xy': 2.0, 'z': 2} # first steps of 3d patch
    GRID_STEPS_2 = {'xy': 1, 'z': 0.1} # second steps of 3d patch
    
    Zscan = PyTorchStereoCorrel(yaml_file=YAML_FILE)

    Zscan.run_grid_diagnostics( {'x': (0, 500), 'y': (0, 400), 'z': (-100,100)}, GRID_STEPS_1)
    Zscan.run_grid_diagnostics( {'x': (0, 500), 'y': (0, 400), 'z': (-100,100)}, GRID_STEPS_2)

    black_factor = 1# 0 to 1 

    for method in method:
        print('=' * 50)
        print(f"\nIniciando processamento  usando o método {method}...")

        for file in files[method][7:]:

            IMAGES_PATH = Path(PATH / method / file)
            print('-' * 50)

            print(f'Processando arquivos: {file}')

            if method in ('laser', 'pattern'):
                N_IMGS_OPTIONS = [5]
                KERNEL_SIZES = [3]
                FILTER_THRESHOLD = 0.8
                FILTER_RADIUS =15.0
                FILTER_MIN_NEIGHBORS = 5
                MASK_BOUNDS = 10 # Limites do desvio padrão dos pontos interpolados nas T imagens

            if method == 'fringe':
                N_IMGS_OPTIONS = [1]
                KERNEL_SIZES = [1]
                PIXEL_PER_FRINGE = 64
                STEPS = 8
                FILTER_THRESHOLD = 0.1
                FILTER_RADIUS = 10.0
                FILTER_MIN_NEIGHBORS = 15
                MASK_BOUNDS = 30 # Limites da modulação dos pontos interpolados
                
            GRID_LIMITS = {'x': (-100, 500), 'y': (-100, 400), 'z': (-dz, dz)}
            
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

            for n_img in N_IMGS_OPTIONS:

                if method != 'fringe':
                    print(f'Carregando {n_img} pares de imagens...')
                    left_imgs_cpu = read_images_from_disk(left_path, left_imgs_list, n_imgs=n_img)
                    left_imgs_cpu = [cv2.convertScaleAbs(img, alpha=black_factor) for img in left_imgs_cpu]
                    right_imgs_cpu = read_images_from_disk(right_path, right_imgs_list, n_imgs=n_img)
                    right_imgs_cpu = [cv2.convertScaleAbs(img, alpha=black_factor) for img in right_imgs_cpu]
                else:
                    # print(f"Carregando {n_img} pares de imagens...")
                    left_imgs_cpu = read_images_from_disk(left_path, left_imgs_list, n_imgs=len(left_imgs_list))
                    left_imgs_cpu = [cv2.convertScaleAbs(img, alpha=black_factor) for img in left_imgs_cpu]
                    right_imgs_cpu = read_images_from_disk(right_path, right_imgs_list, n_imgs=len(right_imgs_list))
                    right_imgs_cpu = [cv2.convertScaleAbs(img, alpha=black_factor) for img in right_imgs_cpu]
                    fringe_imgs_proc = FringeProcess(camera_resolution=left_imgs_cpu[0].shape[::-1], px_f=PIXEL_PER_FRINGE, steps=STEPS)
                    fringe_imgs_proc.images_left = np.moveaxis(np.array(left_imgs_cpu), 0, -1)
                    fringe_imgs_proc.images_right = np.moveaxis(np.array(right_imgs_cpu),0, -1)
                    abs_left, abs_right, mod_left, mod_right = fringe_imgs_proc.calculate_abs_phi_images(visualize=plot)
                    left_imgs_cpu = [abs_left, mod_left]
                    right_imgs_cpu = [abs_right, mod_right]


                for kernel in KERNEL_SIZES:
                    print(f"Iniciando {method} com kernel espacial {kernel}x{kernel}")
                    
                    t_run_start = time.time()
                                    
                    print("Convertendo imagens (CLAHE, Undistort)...")
                    if method == 'fringe':
                        Zscan.convert_images(left_imgs_cpu, right_imgs_cpu, apply_clahe=False, undist=True)
                    else:
                        Zscan.convert_images(left_imgs_cpu, right_imgs_cpu, apply_clahe=True, undist=True)
                    # del left_imgs_cpu, right_imgs_cpu
                        

                    t_preprocessing_done = time.time()
                    print(f"Pré-processamento de imagens concluído em {t_preprocessing_done - t_run_start:.2f} s")

                    print("Construindo grade 3D e iniciando a correlação...")
                #First Process
                    Zscan.points3d(x_lim=GRID_LIMITS['x'], y_lim=GRID_LIMITS['y'], z_lim=GRID_LIMITS['z'],
                                xy_step=GRID_STEPS_1['xy'], z_step=GRID_STEPS_1['z'])
                    
                    xyz_gpu, corr_gpu, _ = Zscan.process_segmented_z(Kx=kernel, Ky=kernel, stride=1, Nz_block_voxels=20, method=method)

                    t_correlation_done = time.time()
                    print(f"1a Correlação concluída em {t_correlation_done - t_preprocessing_done:.2f} s")

                    if xyz_gpu.numel() == 0:
                        print(f"Nenhum ponto retornado pelo processamento para.")
                        continue

                    print(f"Total de pontos brutos: {xyz_gpu.shape[0]}")
                    if method == 'fringe':
                        filter_mask = (corr_gpu < FILTER_THRESHOLD)
                        xyz_filtered_gpu = xyz_gpu[filter_mask]
                        corr_filtered_gpu = corr_gpu[filter_mask]
                        print(f"Pontos com correlação < {FILTER_THRESHOLD} rad: {xyz_filtered_gpu.shape[0]}")
                    else:
                        filter_mask = (corr_gpu > FILTER_THRESHOLD) & (corr_gpu < 1)
                        xyz_filtered_gpu = xyz_gpu[filter_mask]
                        corr_filtered_gpu = corr_gpu[filter_mask]
                        print(f"Pontos com correlação > {FILTER_THRESHOLD*100} %: {xyz_filtered_gpu.shape[0]}")
                    
                    if xyz_filtered_gpu.numel() == 0:
                        print(f"Nenhum ponto filtrado pelo processamento para.")
                        continue

                    if debug:
                        Zscan.plot_3d_points(x=xyz_filtered_gpu[:,0],
                                            y=xyz_filtered_gpu[:,1],
                                            z=xyz_filtered_gpu[:,2],
                                            color=corr_filtered_gpu,
                                            title=f'1a {method} - Pontos 3D: {n_img} imgs {kernel}x{kernel}')

                    if mask_uv:
                        xyz_filtered_gpu, corr_filtered_gpu, interp_filtered = Zscan.mask_uv_points(xyz_filtered_gpu, corr_filtered_gpu, bounds=MASK_BOUNDS, method=method)
                        if debug:
                            Zscan.plot_3d_points(x=xyz_filtered_gpu[:,0],
                                                y=xyz_filtered_gpu[:,1],
                                                z=xyz_filtered_gpu[:,2],
                                                color=interp_filtered,
                                                title=f'1 DEBUG UV MASK {method} - Pontos 3D filtrados UV: {n_img} imgs {kernel}x{kernel}')
                    if mask_std:
                        xyz_filtered_gpu, corr_filtered_gpu, interp_filtered = Zscan.std_mask_points(xyz_filtered_gpu, corr_filtered_gpu, bounds=MASK_BOUNDS, method=method)
                        if debug:
                            Zscan.plot_3d_points(x=xyz_filtered_gpu[:,0],
                                                y=xyz_filtered_gpu[:,1],
                                                z=xyz_filtered_gpu[:,2],
                                                color=interp_filtered,
                                                title=f'1 DEBUG STD MASK\n {method} - Pontos 3D filtrados UV: {n_img} imgs {kernel}x{kernel}')

                    if euclidian_filter:
                        final_xyz_gpu, final_corr_gpu, interp = Zscan.euclidean_filter(xyz_gpu=xyz_filtered_gpu, corr_gpu=corr_filtered_gpu, interp=interp_filtered, 
                                                                    min_neighbors=FILTER_MIN_NEIGHBORS, radius=FILTER_RADIUS)
                    else:
                        final_xyz_gpu, final_corr_gpu, interp = xyz_filtered_gpu, corr_filtered_gpu, interp_filtered
                    print(f'Z mean: {final_xyz_gpu[:,2].cpu().numpy().mean()},\n\
                            Z max: {final_xyz_gpu[:,2].cpu().numpy().max()},\n\
                           Z min: {final_xyz_gpu[:,2].cpu().numpy().min()}')
                    if plot:
                        Zscan.plot_3d_points(x=final_xyz_gpu[:,0],
                                            y=final_xyz_gpu[:,1],
                                            z=final_xyz_gpu[:,2],
                                            color=final_corr_gpu,
                                            title=f'1a {method} - Pontos 3D: {n_img} imgs {kernel}x{kernel}')
                        first_xyz = final_xyz_gpu
                    if final_xyz_gpu.numel() == 0:
                        print(f"Nenhum ponto final retornado pelo processamento para.")
                        continue


            ######## Second Process  
                         
                    Zscan.build_surface_aligned_grid(final_xyz_gpu, dz=0.5, offset=10.0, debug=debug)
                    xyz_gpu, corr_gpu, _ = Zscan.process_segmented_z(Kx=kernel, Ky=kernel, stride=1, Nz_block_voxels=20, method=method)
                    # xyz_gpu = Zscan.z_final(xyz_gpu)
                    t_correlation_done = time.time()
                    print(f"2a Correlação concluída em {t_correlation_done - t_preprocessing_done:.2f} s")

                    if xyz_gpu.numel() == 0:
                        print(f"Nenhum ponto retornado pelo processamento para.")
                        continue

                    print(f"Total de pontos brutos: {xyz_gpu.shape[0]}")
                    if method == 'fringe':
                        filter_mask = (corr_gpu < FILTER_THRESHOLD)
                        xyz_filtered_gpu = xyz_gpu[filter_mask]
                        corr_filtered_gpu = corr_gpu[filter_mask]
                        print(f"Pontos com correlação < {FILTER_THRESHOLD} rad: {xyz_filtered_gpu.shape[0]}")
                    else:
                        filter_mask = (corr_gpu > FILTER_THRESHOLD) & (corr_gpu < 1)
                        print("xyz_gpu shape:", xyz_gpu.shape)
                        print("filter_mask shape:", filter_mask.shape)
                        print("filter_mask dtype:", filter_mask.dtype)
                        xyz_filtered_gpu = xyz_gpu[filter_mask]
                        corr_filtered_gpu = corr_gpu[filter_mask]
                        print(f"Pontos com correlação > {FILTER_THRESHOLD*100} %: {xyz_filtered_gpu.shape[0]}")
                    
                    if xyz_filtered_gpu.numel() == 0:
                        print(f"Nenhum ponto filtrado pelo processamento para.")
                        continue

                    if debug:
                        Zscan.plot_3d_points(x=xyz_filtered_gpu[:,0],
                                            y=xyz_filtered_gpu[:,1],
                                            z=xyz_filtered_gpu[:,2],
                                            color=corr_filtered_gpu,
                                            title=f'2a normal {method} - Pontos 3D: {n_img} imgs {kernel}x{kernel}')

                    if mask_uv:
                        xyz_filtered_gpu, corr_filtered_gpu, interp_filtered = Zscan.mask_uv_points(xyz_filtered_gpu, corr_filtered_gpu, bounds=MASK_BOUNDS, method=method)
                        if debug:
                            Zscan.plot_3d_points(x=xyz_filtered_gpu[:,0],
                                                y=xyz_filtered_gpu[:,1],
                                                z=xyz_filtered_gpu[:,2],
                                                color=interp_filtered,
                                                title=f'2 DEBUG UV MASK {method} - Pontos 3D filtrados UV: {n_img} imgs {kernel}x{kernel}')
                    if mask_std:
                        xyz_filtered_gpu, corr_filtered_gpu, interp_filtered = Zscan.std_mask_points(xyz_filtered_gpu, corr_filtered_gpu, bounds=MASK_BOUNDS, method=method)
                        if debug:
                            Zscan.plot_3d_points(x=xyz_filtered_gpu[:,0],
                                                y=xyz_filtered_gpu[:,1],
                                                z=xyz_filtered_gpu[:,2],
                                                color=interp_filtered,
                                                title=f'2 DEBUG STD MASK\n {method} - Pontos 3D filtrados UV: {n_img} imgs {kernel}x{kernel}')

                    if euclidian_filter:
                        final_xyz_gpu, final_corr_gpu, interp = Zscan.euclidean_filter(xyz_gpu=xyz_filtered_gpu, corr_gpu=corr_filtered_gpu, interp=interp_filtered, 
                                                                    min_neighbors=FILTER_MIN_NEIGHBORS, radius=FILTER_RADIUS)
                    else:
                        final_xyz_gpu, final_corr_gpu, interp = xyz_filtered_gpu, corr_filtered_gpu, interp_filtered
                    print(f'Z mean: {final_xyz_gpu[:,2].cpu().numpy().mean()},\
                            Z max: {final_xyz_gpu[:,2].cpu().numpy().max()},\
                           Z min: {final_xyz_gpu[:,2].cpu().numpy().min()}')
                    if plot:
                        Zscan.plot_3d_points(x=final_xyz_gpu[:,0],
                                            y=final_xyz_gpu[:,1],
                                            z=final_xyz_gpu[:,2],
                                            color=final_corr_gpu,
                                            title=f'2a normal {method} - Pontos 3D: {n_img} imgs {kernel}x{kernel}')
                    
                   

if __name__ == "__main__":
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU detectada: {props.name}, Memória Total: {props.total_memory / (1024**2):.2f} MB")
    else:
        print("GPU não detectada pelo PyTorch. O código será executado na CPU, o que pode ser muito lento.")
    main()