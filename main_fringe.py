import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import os
import time
import cv2
from cupyx.scipy.signal import correlation_lags

from include.FringeTriangulation import InverseTriangulation


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
    # output_path = '{}-{}-correl'.format(time.strftime("%Y%m%d"), images_path.split('/')[-1])
    # os.makedirs(output_path, exist_ok=True)

    t0 = time.time()
    # Load images
    left_imgs_list = sorted(os.listdir(os.path.join(images_path, 'left')))
    right_imgs_list = sorted(os.listdir(os.path.join(images_path, 'right')))
    t1 = time.time()
    print('Open Correlation images: {} s'.format(round(t1 - t0, 2)))
    # Grid search parameters
    n_imgs_v = [1]
    


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
    x_lim = (-200, 500)
    y_lim = (-100, 300)
    z_lim = (-150, 500)

    # Step size for point cloud
    dxyz = (10, 1) #xy step, z step


    for n_img in n_imgs_v:
        path = 'images/fringe'
        Zscan = InverseTriangulation(yaml_file=yaml_file)
        Zscan.left_images = cp.asarray(cv2.imread(os.path.join(path, 'abs_left.png'), cv2.IMREAD_GRAYSCALE), dtype=np.uint64)
        Zscan.right_images = cp.asarray(cv2.imread(os.path.join(path, 'abs_right.png'), cv2.IMREAD_GRAYSCALE), dtype=np.uint64)
        Zscan.left_mask = cp.asarray(cv2.imread(os.path.join(path, 'mask_left.png'), cv2.IMREAD_GRAYSCALE), dtype=np.uint64)
        Zscan.right_mask = cp.asarray(cv2.imread(os.path.join(path, 'mask_right.png'), cv2.IMREAD_GRAYSCALE), dtype=np.uint64)
        # Configurações iniciais
        x_range = (-200, 500)
        y_range = (-150, 500)
        z_range = (-900, 100)

        # Processamento inicial
        grid = Zscan.points3d(xlim=x_range, ylim=y_range, zlim=z_range, xy_step=10, z_step=1)
        Nx, Ny, Nz, _ = grid.shape
        points = grid.reshape(Nx * Ny, Nz, 3)
        points_result_ar = Zscan.fringe_process(points_3d=points, mod_thresh=0.1)
        xyz = points_result_ar.get()
        Zscan.plot_3d_points(xyz[:,0], xyz[:,1], xyz[:,2], title='xyz')

    # np.save(os.path.join(output_path, 'correl.npy'.format(n_img, kernel)), correl_data)



if __name__ == "__main__":
    main()
