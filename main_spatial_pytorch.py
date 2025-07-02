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

def main():
    """Função principal para executar o pipeline de correlação estéreo."""
    
    YAML_FILE = 'cfg/SM3_20250509.yaml'
    IMAGES_PATH = Path('20250513_1505_step10_esferas_d2')
    
    N_IMGS_OPTIONS = [5]
    KERNEL_SIZES = [3]
    
    GRID_LIMITS = {'x': (0, 400), 'y': (-100, 300), 'z': (-200, 200)}
    GRID_STEPS = {'xy': 2.0, 'z': 2.0}
    CORR_THRESHOLD = 0.8
    SPATIAL_FILTER_RADIUS = 10.0
    SPATIAL_FILTER_MIN_NEIGHBORS = 15
    
    current_timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = Path(f'{current_timestamp}-{IMAGES_PATH.name}-pytorch-correl')
    output_path.mkdir(parents=True, exist_ok=True)

    t_start_total = time.time()
    
    try:
        left_imgs_list = sorted([p.name for p in (IMAGES_PATH / 'left').iterdir()])
        right_imgs_list = sorted([p.name for p in (IMAGES_PATH / 'right').iterdir()])
        if not left_imgs_list or not right_imgs_list:
            print(f"Erro: Não foram encontradas imagens em {IMAGES_PATH}")
            return
    except FileNotFoundError:
        print(f"Erro: Diretório de imagens não encontrado: {IMAGES_PATH}")
        return
        
    print(f'Imagens encontradas. Processamento iniciado...')

    def read_images_from_disk(path: Path, images_list: list, n_imgs: int) -> list:
        """Lê um número específico de imagens em escala de cinza de um diretório."""
        return [cv2.imread(str(path / img_name), cv2.IMREAD_GRAYSCALE) for img_name in images_list[:n_imgs]]

    for n_img in N_IMGS_OPTIONS:
        for kernel in KERNEL_SIZES:
            run_key = f"imgs{n_img}_kernel{kernel}"
            print(f'\n======== Iniciando: {run_key} ========')
            
            t_run_start = time.time()

            Zscan = PyTorchStereoCorrel(yaml_file=YAML_FILE)
            
            print(f"Carregando {n_img} pares de imagens...")
            left_imgs_cpu = read_images_from_disk(IMAGES_PATH / 'left', left_imgs_list, n_img)
            right_imgs_cpu = read_images_from_disk(IMAGES_PATH / 'right', right_imgs_list, n_img)
            
            print("Convertendo imagens (CLAHE, Undistort)...")
            Zscan.convert_images(left_imgs_cpu, right_imgs_cpu, apply_clahe=True, undist=True)
            del left_imgs_cpu, right_imgs_cpu

            t_preprocessing_done = time.time()
            print(f"Pré-processamento de imagens concluído em {t_preprocessing_done - t_run_start:.2f} s")

            print("Construindo grade 3D e iniciando a correlação...")
            Zscan.points3d(
                x_lim=GRID_LIMITS['x'], y_lim=GRID_LIMITS['y'], z_lim=GRID_LIMITS['z'],
                xy_step=GRID_STEPS['xy'], z_step=GRID_STEPS['z']
            )
            
            xyz_gpu, corr_gpu, _ = Zscan.process_segmented_z(
                Kx=kernel, Ky=kernel, stride=1, Nz_block_voxels=40
            )

            t_correlation_done = time.time()
            print(f"Correlação concluída em {t_correlation_done - t_preprocessing_done:.2f} s")

            if xyz_gpu.numel() == 0:
                print(f"Nenhum ponto retornado pelo processamento para {run_key}.")
                continue

            raw_output_filename = output_path / f'raw_points_{run_key}.csv'
            save_point_cloud(raw_output_filename, xyz_gpu, corr_gpu)

            print(f"Total de pontos brutos: {xyz_gpu.shape[0]}")
            filter_mask = corr_gpu > CORR_THRESHOLD
            xyz_filtered_gpu = xyz_gpu[filter_mask]
            corr_filtered_gpu = corr_gpu[filter_mask]
            print(f"Pontos com correlação > {CORR_THRESHOLD}: {xyz_filtered_gpu.shape[0]}")
            
            if xyz_filtered_gpu.numel() > 0:
                filtered_output_filename = output_path / f'filtered_points_{run_key}_corr{CORR_THRESHOLD}.csv'
                save_point_cloud(filtered_output_filename, xyz_filtered_gpu, corr_filtered_gpu)
                Zscan.plot_3d_points(
                    xyz_filtered_gpu[:, 0], xyz_filtered_gpu[:, 1], xyz_filtered_gpu[:, 2],
                    color=corr_filtered_gpu,
                    title=f'Pontos Filtrados (Corr > {CORR_THRESHOLD}) - {run_key}'
                )
                
                print("\nAplicando filtro espacial de outliers...")
                final_xyz_gpu, final_corr_gpu = Zscan.filter_sparse_points(
                    xyz_gpu=xyz_filtered_gpu, corr_gpu=corr_filtered_gpu,
                    min_neighbors=SPATIAL_FILTER_MIN_NEIGHBORS, radius=SPATIAL_FILTER_RADIUS
                )
                print(f"Pontos após o filtro espacial: {final_xyz_gpu.shape[0]}")

                if final_xyz_gpu.numel() > 0:
                    final_output_filename = output_path / f'final_points_{run_key}_rad{SPATIAL_FILTER_RADIUS}_neigh{SPATIAL_FILTER_MIN_NEIGHBORS}.csv'
                    save_point_cloud(final_output_filename, final_xyz_gpu, final_corr_gpu)
                    Zscan.plot_3d_points(
                        final_xyz_gpu[:, 0], final_xyz_gpu[:, 1], final_xyz_gpu[:, 2],
                        color=final_corr_gpu,
                        title=f'Nuvem Final com Filtro Espacial - {run_key}'
                    )

            t_run_end = time.time()
            print(f"======== Concluído: {run_key} em {t_run_end - t_run_start:.2f} s ========")

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