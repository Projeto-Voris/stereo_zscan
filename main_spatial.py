import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import os
import time

from cupyx.scipy.signal import correlation_lags

from include.SpatialCorrelation import StereoSpatialCorrelator
from extras.debugger import load_array_from_csv
from extras.project_points import read_images, points3d_cube
from include.InverseTriangulation import InverseTriangulation
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
    yaml_file = '/home/daniel/Pictures/20250513_1205_step10/sm3_params.yaml'
    images_path = '/home/daniel/Pictures/20250513_1205_step10'
    # fringe_image_name = '016.csv'
    t0 = time.time()
    left_imgs_list = sorted(os.listdir(os.path.join(images_path, 'left')))

    right_imgs_list = sorted(os.listdir(os.path.join(images_path, 'right')))
    t1 = time.time()
    print('Open Correlation images: {} s'.format(round(t1 - t0, 2)))

    n_imgs_v = [10, 20, 30]
    kernel_size = [3, 5, 7]

    for n_img in n_imgs_v:
        for kernel in kernel_size:
            print('======== Number of images: {}'.format(n_img))
            print('======== Kernel size: {}'.format(kernel))

            Zscan_old =  InverseTriangulation(yaml_file=yaml_file)
            Zscan = StereoSpatialCorrelator(yaml_file=yaml_file)

            left_imgs = Zscan.read_images(path=os.path.join(images_path,'left'), images_list=left_imgs_list, n_imgs=n_img)
            right_imgs = Zscan.read_images(path=os.path.join(images_path,'right'), images_list=right_imgs_list, n_imgs=n_img)
            Zscan.convert_images(left_imgs=left_imgs, right_imgs=right_imgs, apply_clahe=True, undist=True)
            Zscan_old.convert_images(left_imgs=left_imgs, right_imgs=right_imgs, apply_clahe=True, undist=True)

            # print('Open Correlation images: {}'.format(n_img))
            t2 = time.time()
            # construct 3D points
            Zscan.points3d(x_lim=(0,400), y_lim=(-125,225), z_lim=(-400,400), xy_step=1, z_step=1)
            # 
            # xyz, corr, _, _ = Zscan.run_batch(r_xy=1, stride=2)
            xyz, corr, _, _ = Zscan.run_batch_2(Kx=kernel, Ky=kernel, stride=kernel)
            xyz = cp.asnumpy(xyz[corr > 0.85])
            corr = cp.asnumpy(corr[corr > 0.85])
            filtered_xyz, filtered_corr = Zscan.filter_sparse_points(xyz=xyz, corr=corr, min_neighbors=10, radius=10)

            #xyz = xyz[corr > 0.9]
            print('1st Corrlation time {} s'.format(round(time.time() - t2, 2)))
            Zscan.plot_3d_points(xyz[:,0], xyz[:,1], xyz[:,2], color=corr, title='xyz')
            Zscan.plot_3d_points(filtered_xyz[:,0], filtered_xyz[:,1], filtered_xyz[:,2], color=filtered_corr, title='Filtered xyz')
            # t3 = time.time()
            # xlim = [min(filtered_xyz[:,0]), max(filtered_xyz[:,0])] 
            # ylim = [min(filtered_xyz[:,1]), max(filtered_xyz[:,1])]
            # zlim = [min(filtered_xyz[:,2]), max(filtered_xyz[:,2])]
            # print('X: {} to {} mm'.format(xlim[0], xlim[1]))
            # print('Y: {} to {} mm'.format(ylim[0], ylim[1]))
            # print('Z: {} to {} mm'.format(zlim[0], zlim[1]))
            # zlim[1] = 0
            # Zscan.points3d(xlim, ylim, zlim, xy_step=1, z_step=1)
            # # xyz, corr, _, _ = Zscan.run_batch(r_xy=.5, stride=2)
            # xyz, corr, _, _ = Zscan.run_batch_2(Kx=kernel, Ky=kernel, stride=1)
            # xyz = cp.asnumpy(xyz[corr > 0.9])
            # corr = cp.asnumpy(corr[corr > 0.9])
            # filtered_xyz, filtered_corr = Zscan.filter_sparse_points(xyz=xyz, corr=corr, min_neighbors=10, radius=10)

            # # xyz = xyz[corr > 0.9]
            # # print('Z: {} to {} mm'.format(min(filtered_xyz[:,2]), max(filtered_xyz[:,2])))
            # print('2nd Correlation timel: {} s'.format(round(time.time() - t3, 2)))
            # Zscan.plot_3d_points(xyz[:,0], xyz[:,1], xyz[:,2], color=corr, title='xyz')
            # Zscan.plot_3d_points(filtered_xyz[:,0], filtered_xyz[:,1], filtered_xyz[:,2], color=filtered_corr, title='Filtered xyz')
            # # save_point_cloud('./sm3_calota_{}.csv'.format(n_img), filtered_xyz)



if __name__ == "__main__":
    main()
