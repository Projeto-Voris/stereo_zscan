import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import os
import time
import gc

# Removido, pois correlation_lags não está sendo usado aqui.
# from cupyx.scipy.signal import correlation_lags 

# Certifique-se de que o nome do arquivo importado corresponde ao nome do seu arquivo Python
# Ex: se o arquivo for SpatialCorrelation_Optimization.py (com O maiúsculo)
# from include.SpatialCorrelation_Optimization import StereoTemporalSpatialCorrel
from include.SpatialCorrelation_optimization import StereoTemporalSpatialCorrel


def save_point_cloud(filename, xyz, corr=None, delimiter=','):
    """
    Save a point cloud to a file.

    Parameters:
    ----------
    filename : str
        Name of the output file (e.g., 'point_cloud.csv').
    xyz : np.ndarray
        3D points of shape (N, 3).
    corr : np.ndarray, optional
        Correlation values of shape (N,). If provided, it will be saved as the fourth column.
    delimiter : str, optional
        Delimiter to use in the output file (default is ',').
    """
    # Garante que os dados estejam na CPU (NumPy) antes de salvar
    if isinstance(xyz, cp.ndarray):
        xyz = cp.asnumpy(xyz)
    if corr is not None and isinstance(corr, cp.ndarray):
        corr = cp.asnumpy(corr)

    if corr is not None:
        if corr.ndim == 1:
            corr = corr[:, None] # Garante que corr seja (N,1) para hstack
        # Combine xyz and corr into a single array
        data = np.hstack((xyz, corr))
        header_str = 'x,y,z,corr'
    else:
        data = xyz
        header_str = 'x,y,z'

    # Save to file
    np.savetxt(filename, data, delimiter=delimiter, header=header_str, comments='')
    print(f"Point cloud saved to {filename}")

def main():
    # Paths for yaml file and images
    yaml_file = 'cfg/SM3_20250509.yaml' # Exemplo, ajuste conforme necessário
    images_path = '20250513_1505_step10_plano_d2' # Exemplo, ajuste conforme necessário

    # Output path
    # Usar um timestamp mais completo para evitar conflitos se rodar rápido
    current_timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_folder_name = f'{current_timestamp}-{images_path.split("/")[-1].split(os.sep)[-1]}-correl'
    output_path = output_folder_name
    os.makedirs(output_path, exist_ok=True)

    t_start_total = time.time()
    
    # Load images (apenas lista de nomes, o carregamento real é por iteração)
    try:
        left_imgs_list = sorted(os.listdir(os.path.join(images_path, 'left')))
        right_imgs_list = sorted(os.listdir(os.path.join(images_path, 'right')))
        if not left_imgs_list or not right_imgs_list:
            print(f"Erro: Não foram encontradas imagens em {os.path.join(images_path, 'left')} ou {os.path.join(images_path, 'right')}")
            return
    except FileNotFoundError:
        print(f"Erro: Diretório de imagens não encontrado: {images_path}")
        return
        
    print(f'Imagens encontradas. Processamento iniciado...')
    
    # Grid search parameters (ajustados para teste rápido)
    n_imgs_v = [20]  # Exemplo: [5, 10, 15, 20]
    kernel_size_v = [7] # Exemplo: [3, 5, 7]
    
    # Dictionary to store correlation data of one point
    # (Certifique-se de que o ponto de teste (idx) existe para todos os grids gerados)
    correl_data_all_runs = {} # Armazenar dados de todas as execuções

    # Determine XYZ bounds #(min, max)
    x_lim = (0, 400) 
    y_lim = (-100, 300)
    z_lim = (-200, 200) # Reduzido para Z para testes mais rápidos, ajuste conforme necessário

    # Step size for point cloud
    dxyz = (2.0, 2.0) # Aumentado para testes mais rápidos (xy step, z step)

    for n_img in n_imgs_v:
        for kernel in kernel_size_v:
            run_key = f"imgs{n_img}_kernel{kernel}"
            print(f'\n======== Iniciando: {run_key} ========')
            
            t_run_start = time.time()

            Zscan = StereoTemporalSpatialCorrel(yaml_file=yaml_file)
            
            # Read images to process based on n_img
            print(f"  Carregando {n_img} pares de imagens...")
            left_imgs_cpu = Zscan.read_images(path=os.path.join(images_path,'left'), images_list=left_imgs_list, n_imgs=n_img)
            right_imgs_cpu = Zscan.read_images(path=os.path.join(images_path,'right'), images_list=right_imgs_list, n_imgs=n_img)
            
            print(f"  Convertendo imagens (CLAHE, Undistort)...")
            # Em main_spatial_otimization.py, linha 109:
            Zscan.convert_images(left_imgs_cpu=left_imgs_cpu, right_imgs_cpu=right_imgs_cpu, apply_clahe=True, undist=True)
            # Zscan.convert_images(left_imgs=left_imgs_cpu, right_imgs=right_imgs_cpu, apply_clahe=True, undist=True)
            del left_imgs_cpu, right_imgs_cpu # Liberar memória CPU das imagens brutas
            gc.collect()

            t_preprocessing_done = time.time()
            print(f"  Pré-processamento de imagens concluído em {t_preprocessing_done - t_run_start:.2f} s")

            print(f"  Construindo grade 3D (Pontos)... Limites X: {x_lim}, Y: {y_lim}, Z: {z_lim}. Steps: {dxyz}")
            Zscan.points3d(x_lim=x_lim, y_lim=y_lim, z_lim=z_lim, xy_step=dxyz[0], z_step=dxyz[1])
            
            print(f"  Iniciando correlação espaço-temporal...")
            # NOTA: Se você atualizou os nomes dos parâmetros na classe, use-os aqui.
            # ex: block_size_y_voxels, Nz_block_voxels
            # xyz_cp, corr_cp, corr_all_cp, stdL_cp, stdR_cp = Zscan.process_segmented_y(
            #     Kx=kernel, Ky=kernel, stride=2, block_size_y_voxels=20 
            # )
            xyz_cp, corr_cp, corr_all_cp, stdL_cp, stdR_cp = Zscan.process_segmented_z(
                Kx=kernel, Ky=kernel, stride=1, Nz_block_voxels=40 # Use Nz_block_voxels se renomeou
            )
            # xyz_cp, corr_cp, corr_all_cp, stdL_cp, stdR_cp = Zscan.process(
            #     Kx=kernel, Ky=kernel, stride=4
            # )

            t_correlation_done = time.time()
            print(f"  Correlação concluída em {t_correlation_done - t_preprocessing_done:.2f} s")

            if xyz_cp.shape[0] == 0:
                print(f"  Nenhum ponto retornado pelo processamento para {run_key}. Pulando salvamento e plotagem.")
                continue

            # Converter resultados principais para numpy para análise e salvamento
            xyz_np = cp.asnumpy(xyz_cp)
            corr_np = cp.asnumpy(corr_cp)
            
            # Salvar nuvem de pontos bruta (antes de qualquer filtro de correlação)
            raw_output_filename = os.path.join(output_path, f'raw_points_{run_key}.csv')
            save_point_cloud(raw_output_filename, xyz_np, corr_np)

            # Análise de ponto específico (exemplo)
            # Certifique-se de que as coordenadas do ponto de teste são realistas para o grid gerado.
            # O ponto médio pode não ser um centro de kernel exato devido ao 'stride'.
            target_x = (x_lim[0] + x_lim[1]) / 2
            target_y = (y_lim[0] + y_lim[1]) / 2
            
            # Encontrar o ponto mais próximo no resultado xyz_np
            distances_to_target = np.sqrt((xyz_np[:,0] - target_x)**2 + (xyz_np[:,1] - target_y)**2)
            if distances_to_target.size > 0:
                idx_closest = np.argmin(distances_to_target)
                
                # Obter dados de correlação ao longo de Z para este ponto mais próximo
                if corr_all_cp is not None and corr_all_cp.shape[0] > idx_closest : # corr_all_cp é (Nc, Nz)
                    corr_all_for_idx_np = cp.asnumpy(corr_all_cp[idx_closest, :]) # Pega a linha de corr_all para o ponto
                    correl_data_all_runs[run_key] = {
                        'xyz_at_idx': xyz_np[idx_closest],
                        'corr_at_idx': corr_np[idx_closest],
                        'corr_curve_at_idx': corr_all_for_idx_np
                    }
                    print(f"  Dados do ponto mais próximo de ({target_x:.2f}, {target_y:.2f}):")
                    print(f"    XYZ: {xyz_np[idx_closest]}, Corr: {corr_np[idx_closest]:.4f}")
                    # plt.figure()
                    # plt.plot(cp.asnumpy(Zscan.z_vals), corr_all_for_idx_np) # Supondo que Zscan.z_vals seja o z_vals do último processamento
                    # plt.title(f"Curva de Correlação em Z para ponto {idx_closest} ({run_key})")
                    # plt.xlabel("Z (mm)")
                    # plt.ylabel("Correlação")
                    # plt.savefig(os.path.join(output_path, f'corr_curve_point_{idx_closest}_{run_key}.png'))
                    # plt.close()
            else:
                print("  Não foi possível encontrar o ponto de teste no resultado.")


            # Filtrar por limiar de correlação
            threshold_corr = 0.8
            filter_mask = corr_np > threshold_corr
            xyz_filtered_np = xyz_np[filter_mask]
            corr_filtered_np = corr_np[filter_mask]
            
            print(f"  Total de pontos brutos: {xyz_np.shape[0]}")
            print(f"  Pontos com correlação > {threshold_corr}: {xyz_filtered_np.shape[0]}")

            if xyz_filtered_np.shape[0] > 0:
                # Salvar nuvem de pontos filtrada
                filtered_output_filename = os.path.join(output_path, f'filtered_points_{run_key}_corr{threshold_corr}.csv')
                save_point_cloud(filtered_output_filename, xyz_filtered_np, corr_filtered_np)

                # Plotar nuvem de pontos filtrada
                print(f"  Plotando nuvem de pontos filtrada...")
                Zscan.plot_3d_points(xyz_filtered_np[:,0], xyz_filtered_np[:,1], xyz_filtered_np[:,2], 
                                     color=corr_filtered_np, 
                                     title=f'Pontos Filtrados (Corr > {threshold_corr}) - {run_key}')
            else:
                print(f"  Nenhum ponto após filtrar com correlação > {threshold_corr}.")

            # Opcional: Usar Zscan.filter_sparse_points (remoção de outliers espaciais)
            # Lembre-se que Zscan.filter_sparse_points espera arrays CuPy e retorna arrays CuPy.
            # use_spatial_filter = False
            # if use_spatial_filter and xyz_filtered_np.shape[0] > 0:
            #     print("  Aplicando filtro espacial de outliers...")
            #     # Converter de volta para CuPy para a função da classe
            #     xyz_to_filter_cp = cp.asarray(xyz_filtered_np)
            #     corr_to_filter_cp = cp.asarray(corr_filtered_np)
            #     
            #     # Ajuste os parâmetros min_neighbors e radius conforme necessário
            #     spatially_filtered_xyz_cp, spatially_filtered_corr_cp = Zscan.filter_sparse_points(
            #         xyz=xyz_to_filter_cp, corr=corr_to_filter_cp, min_neighbors=10, radius=5.0 # Valores de exemplo
            #     )
            #     del xyz_to_filter_cp, corr_to_filter_cp # Liberar memória GPU
            #
            #     if spatially_filtered_xyz_cp.shape[0] > 0:
            #         spatially_filtered_output_filename = os.path.join(output_path, f'spatially_filtered_points_{run_key}.csv')
            #         save_point_cloud(spatially_filtered_output_filename, spatially_filtered_xyz_cp, spatially_filtered_corr_cp)
            #         
            #         print(f"  Plotando nuvem de pontos com filtro espacial...")
            #         Zscan.plot_3d_points(spatially_filtered_xyz_cp, None, None, # Passar xyz direto, ou componentes
            #                              color=spatially_filtered_corr_cp, 
            #                              title=f'Pontos com Filtro Espacial - {run_key}')
            #     else:
            #         print("  Nenhum ponto após filtro espacial.")
            #     del spatially_filtered_xyz_cp, spatially_filtered_corr_cp
            
            del Zscan, xyz_cp, corr_cp, corr_all_cp, stdL_cp, stdR_cp 
            del xyz_np, corr_np # , corr_all_np # corr_all_np não é usado globalmente
            del xyz_filtered_np, corr_filtered_np
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()
            
            t_run_end = time.time()
            print(f"======== Concluído: {run_key} em {t_run_end - t_run_start:.2f} s ========")


    # Salvar o dicionário correl_data_all_runs se necessário
    if correl_data_all_runs:
        np.save(os.path.join(output_path, 'correl_curves_data.npy'), correl_data_all_runs)
        print(f"Dados das curvas de correlação salvos em {os.path.join(output_path, 'correl_curves_data.npy')}")

    t_end_total = time.time()
    print(f"\nProcessamento total concluído em {t_end_total - t_start_total:.2f} s.")
    print(f"Resultados salvos em: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    # Adicionar verificação de GPU para feedback imediato
    try:
        free_mem, total_mem = cp.cuda.Device().mem_info
        print(f"GPU detectada. Memória livre: {free_mem / (1024**2):.2f} MB / Total: {total_mem / (1024**2):.2f} MB")
    except Exception as e:
        print(f"Erro ao verificar status da GPU CuPy: {e}")
        print("Certifique-se de que o CuPy está instalado corretamente e que uma GPU compatível está disponível.")
        # exit() # Descomente se quiser sair se a GPU não for detectada

    main()