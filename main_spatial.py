import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import os
import time

from cupyx.scipy.signal import correlation_lags

from include.SpatialCorrelation import StereoSpatialCorrelator
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

        Zscan = StereoSpatialCorrelator(yaml_file=yaml_file)    
        left_imgs = Zscan.read_images(path=os.path.join(images_path,'left'), images_list=left_imgs_list, n_imgs=n_img)
        right_imgs = Zscan.read_images(path=os.path.join(images_path,'right'), images_list=right_imgs_list, n_imgs=n_img)
        Zscan.convert_images(left_imgs=left_imgs, right_imgs=right_imgs, apply_clahe=True, undist=True)
        print('Open Correlation images: {}'.format(n_img))
        t2 = time.time()
        points_3d = Zscan.points3d(x_lim=(0, 100), y_lim=(0, 100), z_lim=(-100, 100), xy_step=5, z_step=1)
        resukt = Zscan.slide_and_correlate(r_xy=0.1, stride=0.1)
        print('3D meshgrid pts: {} mi '.format(points_3d.shape[0] / 1e6))
        print('Create mesgrid pcl: {} s'.format(round(time.time() - t2, 2)))
        t3 = time.time()
        print('Correl {}'.format(round(time.time() - t3, 2)))
        del uv_left, uv_right, spatial_id, spatial_max, std_corr

     
        # Zscan.plot_3d_points(x=correl_points[:, 0], y=correl_points[:, 1], z=correl_points[:, 2], title='3D points')

    # Zscan.save_points(correl_points, filename='./sm4_parede_win7.csv')

    # Inverse Triangulation for correlation
    # Zscan.read_images(left_imgs=left_images, right_imgs=right_images)
    # Zscan.correlation_process(win_size=5, correl_param=(0.3, 0.7), save_points=False, visualize=True)

    # Inverse Triangulation for fringe projection
    # Zscan.read_images(left_imgs=left_image, right_imgs=right_image)
    # z_zcan_points = Zscan.fringe_process(save_points=True, visualize=True)


if __name__ == "__main__":
    main()
