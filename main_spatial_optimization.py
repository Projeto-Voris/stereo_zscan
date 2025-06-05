import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import os
import time
import gc

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
    images_path = '20250513_1505_step10_esferas_d2' # Exemplo, ajuste conforme necessário

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
    n_imgs_v = [5]  # Exemplo: [5, 10, 15, 20]
    kernel_size_v = [3] # Exemplo: [3, 5, 7]
    
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
            Zscan.convert_images(left_imgs_cpu=left_imgs_cpu, right_imgs_cpu=right_imgs_cpu, apply_clahe=True, undist=True)
            del left_imgs_cpu, right_imgs_cpu # Liberar memória CPU das imagens brutas
            gc.collect()

            t_preprocessing_done = time.time()
            print(f"  Pré-processamento de imagens concluído em {t_preprocessing_done - t_run_start:.2f} s")

            print(f"  Construindo grade 3D (Pontos)... Limites X: {x_lim}, Y: {y_lim}, Z: {z_lim}. Steps: {dxyz}")
            Zscan.points3d(x_lim=x_lim, y_lim=y_lim, z_lim=z_lim, xy_step=dxyz[0], z_step=dxyz[1])
            
            print(f"  Iniciando correlação espaço-temporal...")
            xyz_cp, corr_cp, corr_all_cp, stdL_cp, stdR_cp = Zscan.process_segmented_z(
                Kx=kernel, Ky=kernel, stride=1, Nz_block_voxels=40
            )

            t_correlation_done = time.time()
            print(f"  Correlação concluída em {t_correlation_done - t_preprocessing_done:.2f} s")

            # =================================================================
            # INÍCIO DA ANÁLISE DE DIAGNÓSTICO (MÁX/MÍN)
            # =================================================================
            if xyz_cp.shape[0] > 0:
                print("\n--- Análise de Diagnóstico de Correlação ---")
                
                # Encontra o índice do ponto de correlação MÁXIMA
                idx_max = cp.argmax(corr_cp)
                xyz_max = cp.asnumpy(xyz_cp[idx_max])
                corr_max_val = cp.asnumpy(corr_cp[idx_max])
                print(f"  Ponto de Correlação MÁXIMA:")
                print(f"    -> Coordenadas (X,Y,Z): ({xyz_max[0]:.2f}, {xyz_max[1]:.2f}, {xyz_max[2]:.2f})")
                print(f"    -> Valor da Correlação: {corr_max_val:.4f}")

                # Encontra o índice do ponto de correlação MÍNIMA
                idx_min = cp.argmin(corr_cp)
                xyz_min = cp.asnumpy(xyz_cp[idx_min])
                corr_min_val = cp.asnumpy(corr_cp[idx_min])
                print(f"  Ponto de Correlação MÍNIMA:")
                print(f"    -> Coordenadas (X,Y,Z): ({xyz_min[0]:.2f}, {xyz_min[1]:.2f}, {xyz_min[2]:.2f})")
                print(f"    -> Valor da Correlação: {corr_min_val:.4f}")
                print("------------------------------------------\n")
            # =================================================================
            # FIM DA ANÁLISE DE DIAGNÓSTICO
            # =================================================================

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
            target_x = (x_lim[0] + x_lim[1]) / 2
            target_y = (y_lim[0] + y_lim[1]) / 2
            
            distances_to_target = np.sqrt((xyz_np[:,0] - target_x)**2 + (xyz_np[:,1] - target_y)**2)
            if distances_to_target.size > 0:
                idx_closest = np.argmin(distances_to_target)
                
                if corr_all_cp is not None and corr_all_cp.shape[0] > idx_closest :
                    corr_all_for_idx_np = cp.asnumpy(corr_all_cp[idx_closest, :])
                    correl_data_all_runs[run_key] = {
                        'xyz_at_idx': xyz_np[idx_closest],
                        'corr_at_idx': corr_np[idx_closest],
                        'corr_curve_at_idx': corr_all_for_idx_np
                    }
                    print(f"  Dados do ponto mais próximo de ({target_x:.2f}, {target_y:.2f}):")
                    print(f"  XYZ: {xyz_np[idx_closest]}, Corr: {corr_np[idx_closest]:.4f}")
            else:
                print("  Não foi possível encontrar o ponto de teste no resultado.")


            # Filtrar por limiar de correlação
            threshold_corr = 0.9
            filter_mask = corr_np > threshold_corr
            xyz_filtered_np = xyz_np[filter_mask]
            corr_filtered_np = corr_np[filter_mask]
            
            print(f"  Total de pontos brutos: {xyz_np.shape[0]}")
            print(f"  Pontos com correlação > {threshold_corr}: {xyz_filtered_np.shape[0]}")

            if xyz_filtered_np.shape[0] > 0:
                # Salvar nuvem de pontos filtrada por CORRELAÇÃO
                filtered_output_filename = os.path.join(output_path, f'filtered_points_{run_key}_corr{threshold_corr}.csv')
                save_point_cloud(filtered_output_filename, xyz_filtered_np, corr_filtered_np)

                # Plotar nuvem de pontos filtrada por CORRELAÇÃO
                print(f"  Plotando nuvem de pontos filtrada (por correlação)...")
                Zscan.plot_3d_points(xyz_filtered_np[:,0], xyz_filtered_np[:,1], xyz_filtered_np[:,2], 
                                     color=corr_filtered_np, 
                                     title=f'Pontos Filtrados (Corr > {threshold_corr}) - {run_key}')
                
                # =================================================================
                # SEÇÃO DE FILTRO ESPACIAL (OUTLIERS) ADICIONADA
                # =================================================================
                print("\n  Aplicando filtro espacial de outliers (vizinhança)...")
                
                # Converter de volta para CuPy para a função da classe
                xyz_to_filter_cp = cp.asarray(xyz_filtered_np)
                corr_to_filter_cp = cp.asarray(corr_filtered_np)
                
                # --- Ajuste estes parâmetros ---
                # radius: O raio (em mm) da esfera de busca de vizinhos.
                #         Um bom ponto de partida é 2 a 5 vezes o seu xy_step.
                #         Se dxyz=(2.0, 2.0), tente um raio de 5.0 a 10.0.
                raio_busca = 10.0
                # min_neighbors: O número mínimo de vizinhos que um ponto deve ter
                #                dentro do raio para não ser considerado um outlier.
                #                Valores comuns são entre 5 e 20.
                minimo_vizinhos = 15

                spatially_filtered_xyz_cp, spatially_filtered_corr_cp = Zscan.filter_sparse_points(
                    xyz_gpu=xyz_to_filter_cp, corr_gpu=corr_to_filter_cp, # <--- Nomes corretos
                    min_neighbors=minimo_vizinhos, radius=raio_busca
                )   
                
                print(f"  Pontos antes do filtro espacial: {xyz_to_filter_cp.shape[0]}")
                print(f"  Pontos após o filtro espacial: {spatially_filtered_xyz_cp.shape[0]}")

                del xyz_to_filter_cp, corr_to_filter_cp # Liberar memória GPU
                
                if spatially_filtered_xyz_cp.shape[0] > 0:
                    spatially_filtered_output_filename = os.path.join(output_path, f'final_points_{run_key}_rad{raio_busca}_neigh{minimo_vizinhos}.csv')
                    save_point_cloud(spatially_filtered_output_filename, spatially_filtered_xyz_cp, spatially_filtered_corr_cp)
                    
                    print(f"  Plotando nuvem de pontos final (com filtro espacial)...")
                    Zscan.plot_3d_points(spatially_filtered_xyz_cp[:,0], spatially_filtered_xyz_cp[:,1], spatially_filtered_xyz_cp[:,2],
                                         color=spatially_filtered_corr_cp, 
                                         title=f'Nuvem Final com Filtro Espacial - {run_key}')
                else:
                    print("  Nenhum ponto restou após o filtro espacial.")
                
                if 'spatially_filtered_xyz_cp' in locals():
                    del spatially_filtered_xyz_cp, spatially_filtered_corr_cp
                # =================================================================
                # FIM DO FILTRO ESPACIAL
                # =================================================================
                    
            else:
                print(f"  Nenhum ponto após filtrar com correlação > {threshold_corr}.")

            
            del Zscan, xyz_cp, corr_cp, corr_all_cp, stdL_cp, stdR_cp 
            del xyz_np, corr_np
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