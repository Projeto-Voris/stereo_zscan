import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

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

def run_grid_diagnostics(scanner: PyTorchStereoCorrel, limits: dict, steps: dict):
    """
    Executa um diagnóstico de sensibilidade do grid, verificando como os passos
    no espaço 3D se traduzem em movimento de pixels na imagem.
    """
    print("\n[Diagnóstico] Calculando a sensibilidade do grid para os parâmetros atuais...")
    
    x_mid = limits['x'][0] + (limits['x'][1] - limits['x'][0]) / 2
    y_mid = limits['y'][0] + (limits['y'][1] - limits['y'][0]) / 2
    z_mid = limits['z'][0] + (limits['z'][1] - limits['z'][0]) / 2

    p_center = torch.tensor([[x_mid, y_mid, z_mid]], dtype=torch.float32, device=scanner.device)
    p_step_x = torch.tensor([[x_mid + steps['xy'], y_mid, z_mid]], dtype=torch.float32, device=scanner.device)
    p_step_z = torch.tensor([[x_mid, y_mid, z_mid + steps['z']]], dtype=torch.float32, device=scanner.device)

    uv_center = scanner.transform_gcs2ccs(p_center, 'left')
    uv_step_x = scanner.transform_gcs2ccs(p_step_x, 'left')
    uv_step_z = scanner.transform_gcs2ccs(p_step_z, 'left')

    if uv_center.min() > 0 and uv_step_x.min() > 0 and uv_step_z.min() > 0:
        dist_pix_x = torch.linalg.norm(uv_step_x - uv_center).item()
        dist_pix_z = torch.linalg.norm(uv_step_z - uv_center).item()

        print(f"  > Passo XY de {steps['xy']:.1f} mm equivale a um deslocamento de {dist_pix_x:.3f} pixels na imagem.")
        print(f"  > Passo Z de {steps['z']:.1f} mm equivale a um deslocamento de {dist_pix_z:.3f} pixels na imagem.")
    print("-" * 20)

def main():
    """Função principal para executar o pipeline de correlação estéreo."""
    
    YAML_FILE = 'cfg/SM3_20250509.yaml' # Yaml file with camera parameters
    YAML_FILE = 'cfg/SM4.yaml' # Yaml file with camera parameters


    objects = ['calota']#, 'esfera']

    # distances = [700, 850, 1000, 1150, 1300, 1450, 1600, 1750, 1900, 2050]
    distances =[800]
    offset = 800
    dz = 400
    plot = True #plot 3D points of two iterations
    debug = False #plot debug images
    method = [ 'correl', 'spatial']  # 'correl', 'spatial', 'fringe'
    # method = ['fringe']  # 'correl', 'spatial', 'fringe',

    GRID_STEPS_1 = {'xy': 2.0, 'z': 2} # first steps of 3d patch
    GRID_STEPS_2 = {'xy': 1, 'z': 0.1} # second steps of 3d patch
    
    Zscan = PyTorchStereoCorrel(yaml_file=YAML_FILE)
    run_grid_diagnostics(Zscan, {'x': (0, 500), 'y': (0, 400), 'z': (-100,100)}, GRID_STEPS_1)
    run_grid_diagnostics(Zscan, {'x': (0, 500), 'y': (0, 400), 'z': (-100,100)}, GRID_STEPS_2)

    for method in method:
        for obj in objects:
            for dist in distances:
                print('=' * 50)
                print(f"\nIniciando processamento para {obj} a {dist} mm usando o método {method}...")
                # IMAGES_PATH = Path('20250513_1505_step10_calota_d2')
                IMAGES_PATH = Path('{}/{}'.format(method, obj))
                if method == 'spatial':
                    N_IMGS_OPTIONS = [5]
                    KERNEL_SIZES = [3]
                    FILTER_THRESHOLD = 0.6
                    SPATIAL_FILTER_RADIUS =10.0
                    SPATIAL_FILTER_MIN_NEIGHBORS = 5
                    MASK_BOUNDS = 20 # Limites do desvio padrão dos pontos interpolados nas T imagens
                if method == 'correl':
                    N_IMGS_OPTIONS = [15]
                    KERNEL_SIZES = [1]
                    FILTER_THRESHOLD = 0.8
                    SPATIAL_FILTER_RADIUS = 12.0
                    SPATIAL_FILTER_MIN_NEIGHBORS = 5
                    MASK_BOUNDS = 20 # Limites do desvio padrão dos pontos interpolados nas T imagens

                if method == 'fringe':
                    N_IMGS_OPTIONS = [1]
                    KERNEL_SIZES = [1]
                    PIXEL_PER_FRINGE = 64
                    STEPS = 8
                    FILTER_THRESHOLD = 1
                    SPATIAL_FILTER_RADIUS = 10.0
                    SPATIAL_FILTER_MIN_NEIGHBORS = 15
                    MASK_BOUNDS = 30 # Limites da modulação dos pontos interpolados
                
                
                GRID_LIMITS = {'x': (-100, 500), 'y': (-100, 400), 'z': (int(dist) - offset - dz, int(dist) - offset + dz)}

                current_timestamp = time.strftime("%Y%m%d")
                out = Path('{}-{}'.format(current_timestamp, method))
                output_path = out / obj
                output_path.mkdir(parents=True, exist_ok=True)

                t_start_total = time.time()
                
                try:
                    left_path = IMAGES_PATH / 'left'
                    right_path = IMAGES_PATH / 'right'
                    if method == 'fringe':
                        left_path = IMAGES_PATH / 'debug_images/left'
                        right_path = IMAGES_PATH / 'debug_images/right'
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
                    # print(f"Carregando {n_img} pares de imagens...")
                    left_imgs_cpu = read_images_from_disk(left_path, left_imgs_list, n_imgs=len(left_imgs_list))
                    right_imgs_cpu = read_images_from_disk(right_path, right_imgs_list, n_imgs=len(right_imgs_list))
                    fringe_imgs_proc = FringeProcess(camera_resolution=left_imgs_cpu[0].shape[::-1], px_f=PIXEL_PER_FRINGE, steps=STEPS)
                    fringe_imgs_proc.images_left = np.moveaxis(np.array(left_imgs_cpu), 0, -1)
                    fringe_imgs_proc.images_right = np.moveaxis(np.array(right_imgs_cpu),0, -1)
                    abs_left, abs_right, mod_left, mod_right = fringe_imgs_proc.calculate_abs_phi_images(visualize=debug)
                    left_imgs_cpu = [abs_left, mod_left]
                    right_imgs_cpu = [abs_right, mod_right]

                for n_img in N_IMGS_OPTIONS:

                    if method != 'fringe':
                        print(f'Carregando {n_img} pares de imagens...')
                        left_imgs_cpu = read_images_from_disk(left_path, left_imgs_list, n_imgs=n_img)
                        right_imgs_cpu = read_images_from_disk(right_path, right_imgs_list, n_imgs=n_img)


                    for kernel in KERNEL_SIZES:
                        print(f"Iniciando {method} com kernel espacial {kernel}x{kernel}")
                        
                        t_run_start = time.time()
                                        
                        print("Convertendo imagens (CLAHE, Undistort)...")
                        if method == 'fringe':
                            Zscan.convert_images(left_imgs_cpu, right_imgs_cpu, apply_clahe=False, undist=True)
                        else:
                            Zscan.convert_images(left_imgs_cpu, right_imgs_cpu, apply_clahe=True, undist=True)
                        del left_imgs_cpu, right_imgs_cpu
                            

                        t_preprocessing_done = time.time()
                        print(f"Pré-processamento de imagens concluído em {t_preprocessing_done - t_run_start:.2f} s")

                        print("Construindo grade 3D e iniciando a correlação...")
                        Zscan.points3d(x_lim=GRID_LIMITS['x'], y_lim=GRID_LIMITS['y'], z_lim=GRID_LIMITS['z'],
                                    xy_step=GRID_STEPS_1['xy'], z_step=GRID_STEPS_1['z'])
                        
                        xyz_gpu, corr_gpu, _ = Zscan.process_segmented_z(Kx=kernel, Ky=kernel, stride=1, Nz_block_voxels=30, method=method)

                        if plot and debug:
                            Zscan.plot_3d_points(x=xyz_gpu[:,0].cpu().numpy(),
                                                    y=xyz_gpu[:,1].cpu().numpy(),
                                                    z=xyz_gpu[:,2].cpu().numpy(),
                                                    color=corr_gpu.cpu().numpy(),
                                                    title=f'Antes de filtrar {method} - Pontos 3D: {dist} mm {n_img} imgs {kernel}x{kernel}')
                            

                        t_correlation_done = time.time()
                        print(f"Correlação concluída em {t_correlation_done - t_preprocessing_done:.2f} s")

                        if xyz_gpu.numel() == 0:
                            print(f"Nenhum ponto retornado pelo processamento para.")
                            continue

                        # save_point_cloud(output_path / '{}-{}-correl-{}img-{}kernel.csv', xyz_gpu, corr_gpu)

                        print(f"Total de pontos brutos: {xyz_gpu.shape[0]}")
                        if method == 'fringe':
                            filter_mask = corr_gpu < FILTER_THRESHOLD
                        else:
                            filter_mask = corr_gpu > FILTER_THRESHOLD

                        xyz_filtered_gpu = xyz_gpu[filter_mask]
                        corr_filtered_gpu = corr_gpu[filter_mask]
                        xyz_filtered_gpu, corr_filtered_gpu = Zscan.mask_points(xyz_filtered_gpu, corr_filtered_gpu, bounds=MASK_BOUNDS, method=method)

                        print(f"Pontos com correlação > {FILTER_THRESHOLD}: {xyz_filtered_gpu.shape[0]}")
                        final_xyz_gpu, final_corr_gpu = Zscan.filter_sparse_points(
                                xyz_gpu=xyz_filtered_gpu, corr_gpu=corr_filtered_gpu,
                                min_neighbors=SPATIAL_FILTER_MIN_NEIGHBORS, radius=SPATIAL_FILTER_RADIUS
                            )
                        if plot:
                            Zscan.plot_3d_points(x=final_xyz_gpu[:,0].cpu().numpy(),
                                                y=final_xyz_gpu[:,1].cpu().numpy(),
                                                z=final_xyz_gpu[:,2].cpu().numpy(),
                                                color=final_corr_gpu.cpu().numpy(),
                                                title=f'1a {method} - Pontos 3D: {dist} mm {n_img} imgs {kernel}x{kernel}')
                            
                        xlim = torch.min(final_xyz_gpu[:, 0]), torch.max(final_xyz_gpu[:, 0])
                        ylim = torch.min(final_xyz_gpu[:, 1]), torch.max(final_xyz_gpu[:, 1])
                        zlim = torch.min(final_xyz_gpu[:, 2]), torch.max(final_xyz_gpu[:, 2])


                        print("Construindo 2a grade 3D e iniciando a correlação...")
                        Zscan.points3d(x_lim=xlim, y_lim=ylim, z_lim=zlim,
                                    xy_step=GRID_STEPS_2['xy'], z_step=GRID_STEPS_2['z'])
                        
                        xyz_gpu, corr_gpu, _ = Zscan.process_segmented_z(Kx=kernel, Ky=kernel, stride=1, Nz_block_voxels=5, method=method)

                        t_correlation_done = time.time()
                        print(f"2a Correlação concluída em {t_correlation_done - t_preprocessing_done:.2f} s")

                        if xyz_gpu.numel() == 0:
                            print(f"Nenhum ponto retornado pelo processamento para.")
                            continue
                        
                        if xyz_filtered_gpu.numel() > 0:
                            print("\nAplicando filtro espacial de outliers...")
                            final_xyz_gpu, final_corr_gpu = Zscan.filter_sparse_points(
                                xyz_gpu=xyz_filtered_gpu, corr_gpu=corr_filtered_gpu,
                                min_neighbors=SPATIAL_FILTER_MIN_NEIGHBORS, radius=SPATIAL_FILTER_RADIUS
                            )
                            print(f"Pontos após o filtro espacial: {final_xyz_gpu.shape[0]}")
                            
                            if final_xyz_gpu.numel() > 0:
                                save_point_cloud(output_path / '{}-{}-correl-{}img-{}kernel.csv'.format(obj, dist, n_img, kernel),
                                                final_xyz_gpu, final_corr_gpu)

                        t_run_end = time.time()
                        print(f"======== Concluído: em {t_run_end - t_run_start:.2f} s ========")
                        if plot:
                            Zscan.plot_3d_points(x=final_xyz_gpu[:,0].cpu().numpy(),
                                                y=final_xyz_gpu[:,1].cpu().numpy(),
                                                z=final_xyz_gpu[:,2].cpu().numpy(),
                                                color=final_corr_gpu.cpu().numpy(),
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