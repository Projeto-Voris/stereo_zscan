import numpy as np
import matplotlib.pyplot as plt
import os
import time

from cupyx.scipy.signal import correlation_lags

from include.InverseTriangulation import InverseTriangulation
from extras.debugger import load_array_from_csv
from extras.project_points import read_images, points3d_cube


def main():
    yaml_file = 'cfg/SM3_20250203.yaml'
    images_path = '/home/daniel/Insync/daniel.regner@labmetro.ufsc.br/Google Drive - Shared drives/VORIS  - Equipe/Sistema de Medição 3 - Stereo Ativo - Projeção Laser/Imagens/Testes/20250203 - SM3 - speckle'
    Nimg = 40
    method = 'spatial'
    # fringe_image_name = '016.csv'
    t0 = time.time()
    left_images = read_images(os.path.join(images_path, 'left', ),
                              sorted(os.listdir(os.path.join(images_path, 'left'))),
                              n_images=Nimg, visualize=False, CLAHE=True)

    right_images = read_images(os.path.join(images_path, 'right', ),
                               sorted(os.listdir(os.path.join(images_path, 'right'))),
                               n_images=Nimg, visualize=False, CLAHE=True)
    # left_image = load_array_from_csv('csv/l_phasemap.txt', delimiter='\t')
    # right_image = load_array_from_csv('csv/r_phasemap.txt', delimiter='\t')
    t1 = time.time()
    print('Open Correlation images: {} s'.format(round(t1 - t0, 2)))
    # points_3d = points3d_cube(x_lim=(-50, 50), y_lim=(50, 50), z_lim=(-200, 200),
    #                           xy_step=5, z_step=0.1, visualize=False)  # pontos para SM4
    #
    # print("Create meshgrid of {} points: {} s".format(points_3d.shape[0], round(time.time() - t1, 2)))

    n_imgs_v = [10]
    for n_img in n_imgs_v:
        print('Open Correlation images: {} s'.format(n_img))
        t2 = time.time()
        Zscan = InverseTriangulation(yaml_file=yaml_file)
        Zscan.read_images(right_imgs=right_images[:, :, :n_img], left_imgs=left_images[:, :, :n_img])
        points_3d = Zscan.points3d(x_lim=(-600, 600), y_lim=(-400, 400), z_lim=(-200, 200), xy_step=10, z_step=1,
                                   visualize=False)
        print('3D meshgrid pts: {} mi '.format(points_3d.shape[0] / 1e6))
        print('Create mesgrid pcl: {} s'.format(round(time.time() - t2, 2)))
        t3 = time.time()
        if method == 'spatial':

            correl_points = Zscan.spat_temp_correl_process(points_3d=points_3d, visualize=True, save_points=False,
                                                           win_size=21, threshold=0.7)
            print('First Correl {}'.format(round(time.time() - t3, 2)))
            t4 = time.time()

            xlim, ylim, zlim = [min(correl_points[:, 0]), max(correl_points[:, 0])], [min(correl_points[:, 1]),
                                                                                      max(correl_points[:, 1])], [
                min(correl_points[:, 2]), max(correl_points[:, 2])]
            points_3d_2 = Zscan.points3d(x_lim=xlim, y_lim=ylim, z_lim=zlim, z_step=.5, xy_step=1, visualize=True)

            print('2nd 3D meshgrid pts: {} mi'.format(points_3d_2.shape[0] / 1e6))
            print('Create second meshgrid pcl: {} s'.format(round(time.time() - t4, 2)))
            t4 = time.time()
            correl_points = Zscan.spat_temp_correl_process(points_3d=points_3d_2, visualize=True, save_points=False,
                                                           win_size=15, threshold=0.9)
            print('Second Correl {}'.format(round(time.time() - t4, 2)))
        else:

            correl_points = Zscan.temp_correlation_process(points_3d=points_3d, visualize=True, save_points=False, threshold=0.9)
            print('First Correl {}'.format(round(time.time() - t3, 2)))
            t4 = time.time()

            xlim, ylim, zlim = [min(correl_points[:, 0]), max(correl_points[:, 0])], [min(correl_points[:, 1]),
                                                                                      max(correl_points[:, 1])], [
                min(correl_points[:, 2]), max(correl_points[:, 2])]
            points_3d_2 = Zscan.points3d(x_lim=xlim, y_lim=ylim, z_lim=zlim, z_step=1, xy_step=.1, visualize=True)

            print('2nd 3D meshgrid pts: {} mi'.format(points_3d_2.shape[0] / 1e6))
            print('Create second meshgrid pcl: {} s'.format(round(time.time() - t4, 2)))
            t4 = time.time()
            correl_points = Zscan.temp_correlation_process(points_3d=points_3d_2, visualize=False, save_points=False, threshold=0.95)
            print('Second Correl {}'.format(round(time.time() - t4, 2)))

    print('Full time: {} s'.format(round(time.time() - t0, 2)))

    # Zscan.save_points(correl_points, filename='./sm4_parede_win7.csv')

    # Inverse Triangulation for correlation
    # Zscan.read_images(left_imgs=left_images, right_imgs=right_images)
    # Zscan.correlation_process(win_size=5, correl_param=(0.3, 0.7), save_points=False, visualize=True)

    # Inverse Triangulation for fringe projection
    # Zscan.read_images(left_imgs=left_image, right_imgs=right_image)
    # z_zcan_points = Zscan.fringe_process(save_points=True, visualize=True)


if __name__ == "__main__":
    main()
