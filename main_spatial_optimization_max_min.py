import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import os
import time
import gc

# Certifique-se de que o nome do arquivo importado está correto
from include.SpatialCorrelation_optimization import StereoTemporalSpatialCorrel


def save_point_cloud(filename, xyz, corr=None, delimiter=','):
    """
    Salva uma nuvem de pontos para um arquivo.
    """
    # Garante que os dados estejam na CPU (NumPy) antes de salvar
    if isinstance(xyz, cp.ndarray):
        xyz = cp.asnumpy(xyz)
    if corr is not None and isinstance(corr, cp.ndarray):
        corr = cp.asnumpy(corr)

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
    # --- Configurações Iniciais ---
    yaml_file = 'cfg/SM3_20250509.yaml'
    images_path = '20250513_1505_step10_plano_d2'
    
    current_timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_folder_name = f'{current_timestamp}-optimization'
    output_path = output_folder_name
    os.makedirs(output_path, exist_ok=True)

    t_start_total = time.time()
    
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
    
    # --- Parâmetros Fixos para a Execução Principal ---
    # Para este exemplo, fixamos n_img e kernel. A otimização será nos filtros.
    n_img = 5
    kernel = 3
    
    x_lim = (0, 400)
    y_lim = (-100, 300)
    z_lim = (-200, 200)
    dxyz = (2.0, 2.0)

    # --- Execução Única da Etapa Cara (Correlação) ---
    print(f"\n======== Executando a Análise Principal (n_img={n_img}, kernel={kernel}) ========")
    Zscan = StereoTemporalSpatialCorrel(yaml_file=yaml_file)
    
    print(f"  Carregando {n_img} pares de imagens...")
    left_imgs_cpu = Zscan.read_images(path=os.path.join(images_path,'left'), images_list=left_imgs_list, n_imgs=n_img)
    right_imgs_cpu = Zscan.read_images(path=os.path.join(images_path,'right'), images_list=right_imgs_list, n_imgs=n_img)
    
    print("  Convertendo imagens (CLAHE, Undistort)...")
    Zscan.convert_images(left_imgs_cpu=left_imgs_cpu, right_imgs_cpu=right_imgs_cpu, apply_clahe=True, undist=True)
    del left_imgs_cpu, right_imgs_cpu
    gc.collect()

    print(f"  Construindo grade 3D (Pontos)...")
    Zscan.points3d(x_lim=x_lim, y_lim=y_lim, z_lim=z_lim, xy_step=dxyz[0], z_step=dxyz[1])
    
    print(f"  Iniciando correlação espaço-temporal (Esta é a parte demorada)...")
    xyz_cp, corr_cp, _, _, _ = Zscan.process_segmented_z(
        Kx=kernel, Ky=kernel, stride=1, Nz_block_voxels=40
    )
    print("  Correlação principal concluída.")

    # Mover os resultados brutos para a CPU para o loop de otimização
    xyz_np_raw = cp.asnumpy(xyz_cp)
    corr_np_raw = cp.asnumpy(corr_cp)
    del xyz_cp, corr_cp
    gc.collect()

    # <<< INÍCIO DO LOOP DE OTIMIZAÇÃO DE PARÂMETROS >>>
    # =================================================================
    print("\n======== INICIANDO LOOP DE OTIMIZAÇÃO DE PARÂMETROS DE FILTRAGEM ========")

    # 1. DEFINIR ESPAÇO DE BUSCA PARA OS PARÂMETROS QUE VOCÊ QUER TESTAR
    # Sinta-se à vontade para alterar estes valores
    search_space = {
        'thresholds_corr': [0.80, 0.85, 0.90, 0.95],
        'min_neighbors_list': [10, 15, 20, 25],
        'radius_list': [10.0] # Raio de busca fixo para simplificar, mas pode ser uma lista também
    }

    # Variáveis para guardar o melhor resultado encontrado
    best_score = -1
    best_params = {}
    best_point_cloud = {}

    # 2. IMPLEMENTAR ESTRATÉGIA DE BUSCA (GRID SEARCH)
    for t_corr in search_space['thresholds_corr']:
        for m_neigh in search_space['min_neighbors_list']:
            for r_busca in search_space['radius_list']:
                
                params_key = f"t_corr={t_corr}, m_neigh={m_neigh}, r_busca={r_busca}"
                print(f"\n--- Testando: {params_key} ---")
                
                # Etapa 1: Aplicar o filtro de limiar de correlação
                filter_mask_corr = corr_np_raw > t_corr
                xyz_after_corr = xyz_np_raw[filter_mask_corr]
                corr_after_corr = corr_np_raw[filter_mask_corr]

                if xyz_after_corr.shape[0] < m_neigh:
                    print(f"  Pontos insuficientes ({xyz_after_corr.shape[0]}) após filtro de correlação. Pulando.")
                    continue

                # Etapa 2: Aplicar o filtro espacial de outliers
                xyz_to_filter_cp = cp.asarray(xyz_after_corr)
                corr_to_filter_cp = cp.asarray(corr_after_corr)
                
                final_xyz_cp, final_corr_cp = Zscan.filter_sparse_points(
                    xyz_gpu=xyz_to_filter_cp, corr_gpu=corr_to_filter_cp,
                    min_neighbors=m_neigh, radius=r_busca
                )
                
                # 3. CALCULAR A MÉTRICA DE QUALIDADE
                num_final_points = final_xyz_cp.shape[0]
                
                if num_final_points > 1: # Precisa de mais de 1 ponto para ter média
                    avg_corr_final = float(cp.mean(final_corr_cp))
                    # Nossa métrica: número de pontos multiplicado pela correlação média
                    current_score = num_final_points * avg_corr_final
                else:
                    avg_corr_final = 0
                    current_score = 0
                
                print(f"  Resultado: {num_final_points} pontos finais | Média Corr: {avg_corr_final:.4f} | Pontuação: {current_score:.2f}")

                # Etapa 4: Verificar se o resultado atual é o melhor encontrado
                if current_score > best_score:
                    print(f"  >>> NOVO MELHOR RESULTADO ENCONTRADO! <<<")
                    best_score = current_score
                    best_params = {'threshold_corr': t_corr, 'min_neighbors': m_neigh, 'radius': r_busca}
                    # Guarda a melhor nuvem de pontos (já na CPU)
                    best_point_cloud = {'xyz': cp.asnumpy(final_xyz_cp), 'corr': cp.asnumpy(final_corr_cp)}

                del final_xyz_cp, final_corr_cp, xyz_to_filter_cp, corr_to_filter_cp
                gc.collect()

    # <<< FIM DO LOOP DE OTIMIZAÇÃO >>>
    # =================================================================

    print("\n\n======== OTIMIZAÇÃO CONCLUÍDA ========")
    if best_score > -1:
        print(f"Melhores parâmetros encontrados: {best_params}")
        print(f"Melhor pontuação de qualidade: {best_score:.2f}")
        
        # Salva e visualiza a MELHOR nuvem de pontos encontrada pelo loop
        final_filename = os.path.join(output_path, f'best_optimized_cloud_score{best_score:.0f}.csv')
        save_point_cloud(final_filename, best_point_cloud['xyz'], best_point_cloud['corr'])
        
        Zscan.plot_3d_points(
            best_point_cloud['xyz'][:,0], 
            best_point_cloud['xyz'][:,1], 
            best_point_cloud['xyz'][:,2], 
            color=best_point_cloud['corr'], 
            title=f'Melhor Nuvem de Pontos Otimizada\n{best_params}'
        )
    else:
        print("Nenhuma combinação de parâmetros produziu um resultado válido.")

    t_end_total = time.time()
    print(f"\nProcessamento total (com otimização) concluído em {(t_end_total - t_start_total)/60:.2f} minutos.")
    print(f"Resultados salvos em: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    try:
        free_mem, total_mem = cp.cuda.Device().mem_info
        print(f"GPU detectada. Memória livre: {free_mem / (1024**2):.2f} MB / Total: {total_mem / (1024**2):.2f} MB")
    except Exception as e:
        print(f"Erro ao verificar status da GPU CuPy: {e}")
        print("Certifique-se de que o CuPy está instalado corretamente e que uma GPU compatível está disponível.")

    main()