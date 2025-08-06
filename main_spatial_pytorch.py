import numpy as np
import torch
import time
import cv2
from pathlib import Path
from typing import Optional

from include.SpatialCorrelation_pytorch import PyTorchStereoCorrel

def save_point_cloud(
    filename: str | Path,
    xyz: torch.Tensor,
    corr: Optional[torch.Tensor] = None,
    delimiter: str = ','
):
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
        
        if dist_pix_x < 0.5 or dist_pix_z < 0.5:
            print("  > [AVISO] O passo do grid parece ser MUITO PEQUENO. Considere aumentar os passos do grid.")
        elif dist_pix_x > 5.0:
            print("  > [AVISO] O passo do grid pode ser grande, causando perda de correlação espacial.")
        else:
            print("  > [INFO] A sensibilidade do grid parece estar em uma faixa razoável.")
    else:
        print("  > [ERRO DE DIAGNÓSTICO] O ponto central do ROI não pôde ser projetado na câmera. Verifique os limites.")
    print("-" * 20)

def main():
    """Função principal para executar o pipeline de correlação estéreo."""
    
    YAML_FILE = 'cfg/SM4.yaml'

    # objects = ['esfera', 'plano']
    objects = ['plano', 'esfera']

    distances = ['700', '850', '1000', '1150', '1300', '1450']#, '1600', '1750', '1900', '2050']
    # distances = ['1600', '1750', '1900', '2050']
    offset = 800
    dz = 250

    for obj in objects:
        for dist in distances:
            
            
            IMAGES_PATH = Path('correl/{}/{}'.format(obj, dist))
            N_IMGS_OPTIONS = [5]
            KERNEL_SIZES = [3]
            GRID_LIMITS = {'x': (0, 500), 'y': (-0, 300), 'z': (int(dist) - offset - dz, int(dist) - offset + dz)}
            GRID_STEPS = {'xy': 1.0, 'z': 0.1}
            CORR_THRESHOLD = 0.8
            SPATIAL_FILTER_RADIUS = 10.0
            SPATIAL_FILTER_MIN_NEIGHBORS = 15
            
            # current_timestamp = time.strftime("%Y%m%d_%H%M%S")
            # output_path = Path('{}-{}-correl-img{}-{}kernel'.format(obj, dist, N_IMGS_OPTIONS[0], KERNEL_SIZES[0]))
            out = Path('correl')
            output_path = out / 'results'
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

            for n_img in N_IMGS_OPTIONS:
                for kernel in KERNEL_SIZES:
                    run_key = f"imgs{n_img}_kernel{kernel}"
                    print(f'\n======== Iniciando: {run_key} ========')
                    
                    t_run_start = time.time()

                    Zscan = PyTorchStereoCorrel(yaml_file=YAML_FILE)
                    
                    run_grid_diagnostics(Zscan, GRID_LIMITS, GRID_STEPS)

                    print(f"Carregando {n_img} pares de imagens...")
                    left_imgs_cpu = read_images_from_disk(left_path, left_imgs_list, n_img)
                    right_imgs_cpu = read_images_from_disk(right_path, right_imgs_list, n_img)
                    
                    print("Convertendo imagens (CLAHE, Undistort)...")
                    Zscan.convert_images(left_imgs_cpu, right_imgs_cpu, apply_clahe=True, undist=True)
                    del left_imgs_cpu, right_imgs_cpu

                    t_preprocessing_done = time.time()
                    print(f"Pré-processamento de imagens concluído em {t_preprocessing_done - t_run_start:.2f} s")

                    print("Construindo grade 3D e iniciando a correlação...")
                    Zscan.points3d(x_lim=GRID_LIMITS['x'], y_lim=GRID_LIMITS['y'], z_lim=GRID_LIMITS['z'],
                                xy_step=GRID_STEPS['xy'], z_step=GRID_STEPS['z'])
                    
                    xyz_gpu, corr_gpu, _ = Zscan.process_segmented_z(
                        Kx=kernel, Ky=kernel, stride=1, Nz_block_voxels=40
                    )

                    t_correlation_done = time.time()
                    print(f"Correlação concluída em {t_correlation_done - t_preprocessing_done:.2f} s")

                    if xyz_gpu.numel() == 0:
                        print(f"Nenhum ponto retornado pelo processamento para {run_key}.")
                        continue

                    # save_point_cloud(output_path / '{}-{}-correl-{}img-{}kernel.csv', xyz_gpu, corr_gpu)

                    print(f"Total de pontos brutos: {xyz_gpu.shape[0]}")
                    filter_mask = corr_gpu > CORR_THRESHOLD
                    xyz_filtered_gpu = xyz_gpu[filter_mask]
                    corr_filtered_gpu = corr_gpu[filter_mask]
                    print(f"Pontos com correlação > {CORR_THRESHOLD}: {xyz_filtered_gpu.shape[0]}")
                    
                    if xyz_filtered_gpu.numel() > 0:
                        # save_point_cloud(output_path / f'filtered_points_{run_key}_corr{CORR_THRESHOLD}.csv', 
                                        # xyz_filtered_gpu, corr_filtered_gpu)

                        print("\nAplicando filtro espacial de outliers...")
                        final_xyz_gpu, final_corr_gpu = Zscan.filter_sparse_points(
                            xyz_gpu=xyz_filtered_gpu, corr_gpu=corr_filtered_gpu,
                            min_neighbors=SPATIAL_FILTER_MIN_NEIGHBORS, radius=SPATIAL_FILTER_RADIUS
                        )
                        print(f"Pontos após o filtro espacial: {final_xyz_gpu.shape[0]}")
                        
                        if final_xyz_gpu.numel() > 0:
                            save_point_cloud(output_path / '{}-{}-correl-{}img-{}kernel.csv'.format(obj, dist, N_IMGS_OPTIONS[0], KERNEL_SIZES[0]),
                                            final_xyz_gpu, final_corr_gpu)

                    t_run_end = time.time()
                    print(f"======== Concluído: {run_key} em {t_run_end - t_run_start:.2f} s ========")
                    # Zscan.plot_3d_points(x=final_xyz_gpu[:,0].cpu().numpy(),
                    #                     y=final_xyz_gpu[:,1].cpu().numpy(),
                    #                     z=final_xyz_gpu[:,2].cpu().numpy(),
                    #                     color=final_corr_gpu.cpu().numpy(),
                    #                     title=f'Pontos 3D - {run_key}')

            t_end_total = time.time()
            print(f"\nProcessamento total concluído em {t_end_total - t_start_total:.2f} s.")
            print(f"Resultados salvos em: {output_path.resolve()}")

if __name__ == "__main__":
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU detectada: {props.name}, Memória Total: {props.total_memory / (1024**2):.2f} MB")
    else:
        print("GPU não detectada pelo PyTorch. O código será executado na CPU, o que pode ser muito lento.")
    main()