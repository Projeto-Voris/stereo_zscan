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
    if corr is not None:
        # Combine xyz and corr into a single array
        data = np.hstack((xyz, corr[:, None]))
    else:
        data = xyz

    # Save to file
    np.savetxt(filename, data, delimiter=delimiter, header='x,y,z,corr' if corr is not None else 'x,y,z', comments='')
    print(f"Point cloud saved to {filename}")

def main():
    # --- Configurações Iniciais ---
    yaml_file = 'cfg/SM4.yaml'
    # images_path = '/home/daniel/Insync/daniel.regner@labmetro.ufsc.br/Google Drive - Shared drives/VORIS - Media/Experimentos/SM3 - Padrão aleatório/2025 IMEKO - Imagens/20250513_1505_step10_plano_d2'
    images_path = 'correl/calota'

    
    # Output path
    output_path = '{}-{}-correl'.format(time.strftime("%Y%m%d"), images_path.split('/')[-1])
    os.makedirs(output_path, exist_ok=True)

    try:
        left_imgs_list = sorted(os.listdir(os.path.join(images_path, 'left')))
        right_imgs_list = sorted(os.listdir(os.path.join(images_path, 'right')))
        if not left_imgs_list or not right_imgs_list:
            print(f"Erro: Não foram encontradas imagens em {os.path.join(images_path, 'left')} ou {os.path.join(images_path, 'right')}")
            return
    except FileNotFoundError:
        print(f"Erro: Diretório de imagens não encontrado: {images_path}")
        return
    


    n_imgs_v = [5]
    kernel_size = [1]

    for n_img in n_imgs_v:
        for kernel in kernel_size:
            print('======== Number of images: {}'.format(n_img))
            print('======== Kernel size: {}'.format(kernel))

            x_lim = (0, 4) 
            y_lim = (0, 4)
            z_lim = (-2, 2)
            dxyz = (2.0, 2.0)
            # --- Execução Única da Etapa Cara (Correlação) ---
            t0 = time.time()
            print(f"\n======== Executando a Análise Principal (n_img={n_img}, kernel={kernel}) ========")
            Zscan = StereoTemporalSpatialCorrel(yaml_file=yaml_file)
            
            print(f"  Carregando {n_img} pares de imagens...")
            left_imgs_cpu = Zscan.read_images(path=os.path.join(images_path,'left'), images_list=left_imgs_list, n_imgs=n_img)
            right_imgs_cpu = Zscan.read_images(path=os.path.join(images_path,'right'), images_list=right_imgs_list, n_imgs=n_img)
            
            print("  Convertendo imagens")
            Zscan.convert_images(left_imgs_cpu=left_imgs_cpu, right_imgs_cpu=right_imgs_cpu, apply_clahe=False, undist=True)
            del left_imgs_cpu, right_imgs_cpu
            gc.collect()

            print(f"  Construindo grade 3D")
            Zscan.points3d(x_lim=x_lim, y_lim=y_lim, z_lim=z_lim, xy_step=dxyz[0], z_step=dxyz[1])
            
            print(f"  Iniciando correlação espaço-temporal")
            xyz_cp, corr_cp, _, _, _ = Zscan.process_segmented_z(Kx=kernel, Ky=kernel, stride=1, Nz_block_voxels=40)
            print("  Correlação principal concluída.")

            print('------- Primeira correlação: {} s'.format(round(time.time() - t0, 2)))
            # Mover os resultados brutos para a CPU para o loop de otimização
            xyz_np_raw = cp.asnumpy(xyz_cp[corr_cp > 0.9])
            corr_np_raw = cp.asnumpy(corr_cp[corr_cp > 0.9])
            del xyz_cp, corr_cp
            gc.collect()
            # Filtrando pontos
            filtered_xyz_cp, final_corr_cp = Zscan.filter_sparse_points(xyz=xyz_np_raw, corr=corr_np_raw, min_neighbors=5, radius=5)
            xyz = cp.asnumpy(filtered_xyz_cp)
            corr = cp.asnumpy(final_corr_cp)
            Zscan.plot_3d_points(xyz[:,0], xyz[:,1], xyz[:,2], color=corr, title='xyz')

            xlim=(min(cp.asnumpy(filtered_xyz_cp[:,0])), max(cp.asnumpy(filtered_xyz_cp[:,0])))
            ylim=(min(cp.asnumpy(filtered_xyz_cp[:,1])), max(cp.asnumpy(filtered_xyz_cp[:,1])))
            zlim=(min(cp.asnumpy(filtered_xyz_cp[:,2])), max(cp.asnumpy(filtered_xyz_cp[:,2])))
            print('  Limites XYZ: x={} y={} z={}'.format(xlim, ylim, zlim))

            t0 = time.time()
            print(f"  Construindo segunda grade 3D (Pontos)...")
            Zscan.points3d(x_lim=x_lim, y_lim=y_lim, z_lim=z_lim, xy_step=1, z_step=1)
            
            print(f"  Iniciando correlação espaço-temporal")
            xyz_cp, corr_cp, _, _, _ = Zscan.process_segmented_z(Kx=kernel, Ky=kernel, stride=1, Nz_block_voxels=40)

            # Mover os resultados brutos para a CPU para o loop de otimização
            xyz_np_raw = cp.asnumpy(xyz_cp[corr_cp > 0.9])
            corr_np_raw = cp.asnumpy(corr_cp[corr_cp > 0.9])
            del xyz_cp, corr_cp
            gc.collect()
            filtered_xyz_cp, final_corr_cp = Zscan.filter_sparse_points(xyz=xyz_np_raw, corr=corr_np_raw,min_neighbors=10, radius=5)
            xyz = cp.asnumpy(filtered_xyz_cp)
            corr = cp.asnumpy(final_corr_cp)
            print('------- Segunda correlação: {} s'.format(round(time.time() - t0, 2)))

            # filtered_xyz, filtered_corr = Zscan.filter_sparse_points(xyz=xyz, corr=corr, min_neighbors=5, radius=10)
            # np.savetxt(os.path.join(output_path, 'correl_filtered_imgs{}_kernel{}.txt'.format(n_img, kernel, n_img)), filtered_xyz, delimiter=',')
            Zscan.plot_3d_points(xyz[:,0], xyz[:,1], xyz[:,2], color=corr, title='xyz')
   
    # np.save(os.path.join(output_path, 'correl.npy'.format(n_img, kernel)), correl_data)



if __name__ == "__main__":
    main()
