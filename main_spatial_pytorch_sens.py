import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import time
import cv2

from include.SpatialCorrelation_pytorch import PyTorchStereoCorrel

def save_point_cloud(filename, xyz, corr=None, delimiter=','):
    # Garante que os tensores PyTorch sejam movidos para a CPU e convertidos para NumPy
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
    # Paths
    yaml_file = 'cfg/SM3_20250509.yaml'
    images_path = '20250513_1505_step10_esferas_d2'
    
    current_timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_folder_name = f'{current_timestamp}-{os.path.basename(images_path)}-pytorch-correl'
    output_path = output_folder_name
    os.makedirs(output_path, exist_ok=True)

    t_start_total = time.time()
    
    try:
        left_imgs_list = sorted(os.listdir(os.path.join(images_path, 'left')))
        right_imgs_list = sorted(os.listdir(os.path.join(images_path, 'right')))
        if not left_imgs_list or not right_imgs_list:
            print(f"Erro: Não foram encontradas imagens em {images_path}")
            return
    except FileNotFoundError:
        print(f"Erro: Diretório de imagens não encontrado: {images_path}")
        return
        
    print(f'Imagens encontradas. Processamento iniciado...')
    
    # Parâmetros do Grid Search
    n_imgs_v = [5]
    kernel_size_v = [3]
    
    x_lim, y_lim, z_lim = (0, 400), (-100, 300), (-200, 200)
    dxyz = (1.0, 1.0)

    for n_img in n_imgs_v:
        for kernel in kernel_size_v:
            run_key = f"imgs{n_img}_kernel{kernel}"
            print(f'\n======== Iniciando: {run_key} ========')
            
            t_run_start = time.time()

            # Instancia a classe PyTorch
            Zscan = PyTorchStereoCorrel(yaml_file=yaml_file)
            
            # <--- INÍCIO DO BLOCO DE DIAGNÓSTICO --->
            print("\n[Diagnóstico] Calculando a sensibilidade do grid para os parâmetros atuais...")
            
            # Ponto central aproximado do seu volume de interesse (ROI)
            x_mid = x_lim[0] + (x_lim[1] - x_lim[0]) / 2
            y_mid = y_lim[0] + (y_lim[1] - y_lim[0]) / 2
            z_mid = z_lim[0] + (z_lim[1] - z_lim[0]) / 2

            # Crie tensores para os pontos no dispositivo correto
            p_center = torch.tensor([[x_mid, y_mid, z_mid]], dtype=torch.float32, device=Zscan.device)
            p_step_x = torch.tensor([[x_mid + dxyz[0], y_mid, z_mid]], dtype=torch.float32, device=Zscan.device)
            p_step_z = torch.tensor([[x_mid, y_mid, z_mid + dxyz[1]]], dtype=torch.float32, device=Zscan.device)

            # Projete os pontos para a imagem da câmera esquerda (ou direita)
            uv_center = Zscan.transform_gcs2ccs(p_center, 'left')
            uv_step_x = Zscan.transform_gcs2ccs(p_step_x, 'left')
            uv_step_z = Zscan.transform_gcs2ccs(p_step_z, 'left')

            # Calcule a distância em pixels apenas se os pontos forem válidos (projeção > 0)
            if uv_center.min() > 0 and uv_step_x.min() > 0 and uv_step_z.min() > 0:
                dist_pix_x = torch.linalg.norm(uv_step_x - uv_center).item()
                dist_pix_z = torch.linalg.norm(uv_step_z - uv_center).item()

                print(f"  > Passo XY de {dxyz[0]:.1f} mm equivale a um deslocamento de {dist_pix_x:.3f} pixels na imagem.")
                print(f"  > Passo Z de {dxyz[1]:.1f} mm equivale a um deslocamento de {dist_pix_z:.3f} pixels na imagem.")
                
                if dist_pix_x < 0.5 or dist_pix_z < 0.5:
                    print("  > [AVISO] O passo do grid parece ser MUITO PEQUENO. A câmera pode não distinguir pontos 3D vizinhos. Considere aumentar 'dxyz'.")
                elif dist_pix_x > 5.0: # Um valor um pouco maior que o tamanho do kernel
                    print("  > [AVISO] O passo do grid pode ser grande, causando perda de correlação espacial.")
                else:
                    print("  > [INFO] A sensibilidade do grid parece estar em uma faixa razoável.")
            else:
                print("  > [ERRO DE DIAGNÓSTICO] O ponto central do ROI não pôde ser projetado na câmera. Verifique os limites (x_lim, y_lim, z_lim).")
            print("-" * 20)
            # <--- FIM DO BLOCO DE DIAGNÓSTICO --->

            # Carrega imagens do disco
            def read_images_from_disk(path, images_list, n_imgs):
                return [cv2.imread(os.path.join(path, str(img_name)), cv2.IMREAD_GRAYSCALE) for img_name in images_list[0:n_imgs]]

            print(f" Carregando {n_img} pares de imagens...")
            left_imgs_cpu = read_images_from_disk(os.path.join(images_path,'left'), left_imgs_list, n_img)
            right_imgs_cpu = read_images_from_disk(os.path.join(images_path,'right'), right_imgs_list, n_img)
            
            print(f" Convertendo imagens (CLAHE, Undistort)...")
            Zscan.convert_images(left_imgs_cpu, right_imgs_cpu, apply_clahe=True, undist=True)
            del left_imgs_cpu, right_imgs_cpu

            t_preprocessing_done = time.time()
            print(f" Pré-processamento de imagens concluído em {t_preprocessing_done - t_run_start:.2f} s")

            print(f" Construindo grade 3D...")
            Zscan.points3d(x_lim=x_lim, y_lim=y_lim, z_lim=z_lim, xy_step=dxyz[0], z_step=dxyz[1])
            
            print(f" Iniciando correlação espaço-temporal com PyTorch...")
            xyz_gpu, corr_gpu, _, _, _ = Zscan.process_segmented_z(
                Kx=kernel, Ky=kernel, stride=1, Nz_block_voxels=40
            )

            t_correlation_done = time.time()
            print(f" Correlação concluída em {t_correlation_done - t_preprocessing_done:.2f} s")

            if xyz_gpu.numel() == 0:
                print(f" Nenhum ponto retornado pelo processamento para {run_key}.")
                continue

            raw_output_filename = os.path.join(output_path, f'raw_points_{run_key}.csv')
            save_point_cloud(raw_output_filename, xyz_gpu, corr_gpu)

            threshold_corr = 0.8
            filter_mask = corr_gpu > threshold_corr
            xyz_filtered_gpu = xyz_gpu[filter_mask]
            corr_filtered_gpu = corr_gpu[filter_mask]
            
            print(f" Total de pontos brutos: {xyz_gpu.shape[0]}")
            print(f" Pontos com correlação > {threshold_corr}: {xyz_filtered_gpu.shape[0]}")

            if xyz_filtered_gpu.shape[0] > 0:
                filtered_output_filename = os.path.join(output_path, f'filtered_points_{run_key}_corr{threshold_corr}.csv')
                save_point_cloud(filtered_output_filename, xyz_filtered_gpu, corr_filtered_gpu)

                # O plot foi removido para focar na execução, mas pode ser reativado
                # print(f" Plotando nuvem de pontos filtrada (por correlação)...")
                
                print("\n Aplicando filtro espacial de outliers...")
                raio_busca, minimo_vizinhos = 10.0, 15
                
                final_xyz_gpu, final_corr_gpu = Zscan.filter_sparse_points(
                    xyz_gpu=xyz_filtered_gpu, corr_gpu=corr_filtered_gpu,
                    min_neighbors=minimo_vizinhos, radius=raio_busca
                )
                
                print(f" Pontos antes do filtro espacial: {xyz_filtered_gpu.shape[0]}")
                print(f" Pontos após o filtro espacial: {final_xyz_gpu.shape[0]}")
                
                if final_xyz_gpu.shape[0] > 0:
                    final_output_filename = os.path.join(output_path, f'final_points_{run_key}_rad{raio_busca}_neigh{minimo_vizinhos}.csv')
                    save_point_cloud(final_output_filename, final_xyz_gpu, final_corr_gpu)

            t_run_end = time.time()
            print(f"======== Concluído: {run_key} em {t_run_end - t_run_start:.2f} s ========")

    t_end_total = time.time()
    print(f"\nProcessamento total concluído em {t_end_total - t_start_total:.2f} s.")
    print(f"Resultados salvos em: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    if torch.cuda.is_available():
        t = torch.cuda.get_device_properties(0)
        print(f"GPU detectada: {t.name}, Memória Total: {t.total_memory / (1024**2):.2f} MB")
    else:
        print("GPU não detectada pelo PyTorch. O código será executado na CPU, o que pode ser muito lento.")
    main()