import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import os
import time

from cupyx.scipy.signal import correlation_lags

from include.InverseTriangulation import InverseTriangulation
from extras.debugger import load_array_from_csv
from extras.project_points import read_images, points3d_cube


def main():
    yaml_file = 'cfg/20250424.yaml'
    images_path = '/home/daniel/Insync/daniel.regner@labmetro.ufsc.br/Google Drive - Shared drives/VORIS - Media/Experimentos/SM3 - Padrão aleatório/20250425 -Grasshopper f_25mm'
    Nimg = 30
    method = 'spatial'
    # fringe_image_name = '016.csv'
    t0 = time.time()
    left_imgs_list = sorted(os.listdir(os.path.join(images_path, 'left')))

    right_imgs_list = sorted(os.listdir(os.path.join(images_path, 'right')))
    t1 = time.time()
    print('Open Correlation images: {} s'.format(round(t1 - t0, 2)))

    n_imgs_v = [10]

    for n_img in n_imgs_v:

        Zscan = InverseTriangulation(yaml_file=yaml_file)    
        left_imgs = Zscan.read_images(path=os.path.join(images_path,'left'), images_list=left_imgs_list, n_imgs=n_img)
        right_imgs = Zscan.read_images(path=os.path.join(images_path,'right'), images_list=right_imgs_list, n_imgs=n_img)
        Zscan.convert_images(left_imgs=left_imgs, right_imgs=right_imgs, apply_clahe=True, undist=True)
        kernel_pts, kernel_pts_ho = Zscan.correlate_full_space(kernel_size=3)
        print('Open Correlation images: {}'.format(n_img))
        t2 = time.time()
        points_3d = Zscan.points3d(x_lim=(-200, 200), y_lim=(-200, 200), z_lim=(-500, 500), xy_step=5, z_step=1,
                                   visualize=False)
        print('3D meshgrid pts: {} mi '.format(points_3d.shape[0] / 1e6))
        print('Create mesgrid pcl: {} s'.format(round(time.time() - t2, 2)))
        if method == 'spatial':
            t3 = time.time()
            uv_left = Zscan.transform_gcs2ccs(points_3d=points_3d, cam_name='left')
            uv_right = Zscan.transform_gcs2ccs(points_3d=points_3d, cam_name='right')
            spatial_id, spatial_max, std_corr = Zscan.spatial_correl(uv_left=uv_left, uv_right=uv_right, save_points=False)
            correl_mask = Zscan.correl_mask(std_correl=std_corr, correl_max=spatial_max, correl_thresh=0.95, std_thresh=20)
            correl_points =  points_3d[np.asarray(cp.asnumpy(spatial_id[correl_mask])).astype(np.int32)]
            print('Correl {}'.format(round(time.time() - t3, 2)))
            del uv_left, uv_right, spatial_id, spatial_max, std_corr

        if method == 'temporal':
            t = time.time()
            uv_left = Zscan.transform_gcs2ccs(points_3d=points_3d, visualize=False, save_points=False)
            uv_right = Zscan.transform_gcs2ccs(points_3d=points_3d, visualize=False, save_points=False)
            inter_l = Zscan.bi_interpolation(uv_points=uv_left)
            inter_r = Zscan.bi_interpolation(uv_points=uv_right)
            ho, hmax = Zscan.temp_cross_correlation(left_Igray=inter_l, right_Igray=inter_r)
            correl_mask = Zscan.correl_mask(std_correl=ho, correl_max=hmax, correl_thresh=0.9, std_thresh=20)
            correl_points = points_3d[np.asarray(cp.asnumpy(ho[correl_mask])).astype(np.int32)]
            print('Correl {}'.format(round(time.time() - t, 2)))
            del uv_left, uv_right, inter_l, inter_r, ho, hmax
        
        Zscan.plot_3d_points(x=correl_points[:, 0], y=correl_points[:, 1], z=correl_points[:, 2], title='3D points')

    # Zscan.save_points(correl_points, filename='./sm4_parede_win7.csv')

    # Inverse Triangulation for correlation
    # Zscan.read_images(left_imgs=left_images, right_imgs=right_images)
    # Zscan.correlation_process(win_size=5, correl_param=(0.3, 0.7), save_points=False, visualize=True)

    # Inverse Triangulation for fringe projection
    # Zscan.read_images(left_imgs=left_image, right_imgs=right_image)
    # z_zcan_points = Zscan.fringe_process(save_points=True, visualize=True)


if __name__ == "__main__":
    main()
