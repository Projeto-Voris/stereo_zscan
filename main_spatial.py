import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import os
import time

from cupyx.scipy.signal import correlation_lags

from include.SpatialCorrelation import StereoTemporalSpatialCorrel


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
    # Paths for yaml file and images
    yaml_file = 'cfg/SM3_20250509.yaml'
    images_path = '/home/daniel/Insync/daniel.regner@labmetro.ufsc.br/Google Drive - Shared drives/VORIS - Media/Experimentos/SM3 - Padrão aleatório/2025 IMEKO - Imagens/20250513_1505_step10_plano_d2'

    # Output path
    output_path = '{}-{}-correl'.format(time.strftime("%Y%m%d"), images_path.split('/')[-1])
    os.makedirs(output_path, exist_ok=True)

    t0 = time.time()
    # Load images
    left_imgs_list = sorted(os.listdir(os.path.join(images_path, 'left')))
    right_imgs_list = sorted(os.listdir(os.path.join(images_path, 'right')))
    t1 = time.time()
    print('Open Correlation images: {} s'.format(round(t1 - t0, 2)))
    # Grid search parameters
    n_imgs_v = [5,10, 15,20]
    kernel_size = [3, 5, 7]
    
    # Dictionary to store correlation data of one point
    correl_data = {str(n): {str(k): [] for k in kernel_size} for n in n_imgs_v}


    # Determine XYZ bounds #(min, max)
    # D3
    # x_lim = (50, 300) 
    # y_lim = (0, 130)
    # z_lim = (200, 300)
    # D2
    # x_lim = (50, 300) 
    # y_lim = (0, 130)
    # z_lim = (-120, 0)
    # Generic
    x_lim = (0, 400) 
    y_lim = (-100, 300)
    z_lim = (-400, 400)

    # Step size for point cloud
    dxyz = (0.5, 0.5) #xy step, z step
    n_imgs_v = [10]
    kernel_size = [3]

    for n_img in n_imgs_v:
        for kernel in kernel_size:
            print('======== Number of images: {}'.format(n_img))
            print('======== Kernel size: {}'.format(kernel))


            Zscan = StereoTemporalSpatialCorrel(yaml_file=yaml_file)
            # Read images to process based on n_img
            left_imgs = Zscan.read_images(path=os.path.join(images_path,'left'), images_list=left_imgs_list, n_imgs=n_img)
            right_imgs = Zscan.read_images(path=os.path.join(images_path,'right'), images_list=right_imgs_list, n_imgs=n_img)
            Zscan.convert_images(left_imgs=left_imgs, right_imgs=right_imgs, apply_clahe=True, undist=True)

            # print('Open Correlation images: {}'.format(n_img))
            t2 = time.time()
            # construct 3D points
            Zscan.points3d(x_lim=x_lim, y_lim=y_lim, z_lim=z_lim, xy_step=dxyz[0], z_step=dxyz[1])
            
            # Perform correlation
            # xyz, corr, corr_all, _, _ = Zscan.process_segmented_y(Kx=kernel, Ky=kernel, stride=2, block_size_y=10, save_correlation=True)
            xyz, corr, corr_all, _, _ = Zscan.process_segmented_z(Kx=kernel, Ky=kernel, stride=1, Nz_block=40, save_correlation=True)
            # xyz, corr, corr_all, _, _ = Zscan.process(Kx=kernel, Ky=kernel, stride=4, save_correlation=True)

            # Convert to numpy arrays
            print('---- Corrlation time {} s'.format(round(time.time() - t2, 2)))
            xyz = cp.asnumpy(xyz)
            corr = cp.asnumpy(corr)
            corr_all = cp.asnumpy(corr_all)

            # Get index of a specific point (center of the grid)
            idx = np.where((xyz[:,0] == np.mean(x_lim[:2])) & (xyz[:,1] == np.mean(y_lim[:2])))[0]
            print('xyz', xyz[idx])
            print('corr', corr[idx])
            print('xyz shape', xyz.shape)
            print('corr shape', corr.shape)
            print('Max corr', np.max(corr))
            correl_data[str(n_img)][str(kernel)] = corr_all[idx]
            xyz = xyz[corr > 0.8]
            corr = corr[corr > 0.8]
            np.savetxt(os.path.join(output_path, 'correl_imgs{}_kernel{}.txt'.format(n_img, kernel, n_img)), xyz, delimiter=',')
            # filtered_xyz, filtered_corr = Zscan.filter_sparse_points(xyz=xyz, corr=corr, min_neighbors=5, radius=10)
            # np.savetxt(os.path.join(output_path, 'correl_filtered_imgs{}_kernel{}.txt'.format(n_img, kernel, n_img)), filtered_xyz, delimiter=',')
            Zscan.plot_3d_points(xyz[:,0], xyz[:,1], xyz[:,2], color=corr, title='xyz')
   
    # np.save(os.path.join(output_path, 'correl.npy'.format(n_img, kernel)), correl_data)



if __name__ == "__main__":
    main()
