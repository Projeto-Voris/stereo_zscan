import gc

import numpy as np
import os
import time

from include.InverseTriangulation import InverseTriangulation
from extras.project_points import read_images


def main():
    yaml_file = 'cfg/SM4_20241211.yaml'
    images_path = '/home/daniel/Insync/daniel.regner@labmetro.ufsc.br/Google Drive - Shared drives/VORIS  - Equipe/Sistema de Medição 3 - Stereo Ativo - Projeção Laser/Imagens/Testes/SM4-20241211 - cilindro'

    Nimg = 40
    fringe_image_name = '016.csv'
    t0 = time.time()
    left_images = read_images(os.path.join(images_path, 'left', ),
                              sorted(os.listdir(os.path.join(images_path, 'left'))),
                              n_images=Nimg, visualize=False, CLAHE=True)

    right_images = read_images(os.path.join(images_path, 'right', ),
                               sorted(os.listdir(os.path.join(images_path, 'right'))),
                               n_images=Nimg, visualize=False, CLAHE=True)
    # left_image = load_array_from_csv('csv/l_phasemap.txt', delimiter='\t')
    # right_image = load_array_from_csv('csv/r_phasemap.txt', delimiter='\t')
    print('Open images: {} s'.format(round(time.time() - t0, 2)))
    # points_3d = points3d_cube(x_lim=(-50, 50), y_lim=(50, 50), z_lim=(-200, 200),
    #                           xy_step=5, z_step=0.1, visualize=False)  # pontos para SM4
    #
    # print("Create meshgrid of {} points: {} s".format(points_3d.shape[0], round(time.time() - t1, 2)))

    n_imgs_v = [10]
    for n_img in n_imgs_v:
        t2 = time.time()
        Zscan = InverseTriangulation(yaml_file=yaml_file)
        Zscan.read_images(right_imgs=right_images[:, :, :n_img], left_imgs=left_images[:, :, :n_img])
        points_3d = Zscan.points3d_gpu(x_lim=(-600, 600), y_lim=(-600, 600), z_lim=(-800, 800), xy_step=15, z_step=.5,
                                       visualize=False)
        print('3D meshgrid pts: {} mi '.format(points_3d.shape[0] / 1e6))
        print('Create mesgrid pcl: {} s'.format(round(time.time() - t2, 2)))
        t3 = time.time()

        correl_points = Zscan.correlation_process(points_3d=points_3d, visualize=True, save_points=False,
                                                  win_size=7, threshold=0.9, method='spatial')
        print('First Correl {}'.format(round(time.time() - t3, 2)))
        t4 = time.time()

        xlim = tuple(float(x) for x in [min(correl_points[:, 0]), max(correl_points[:, 0])])
        ylim = tuple(float(x) for x in [min(correl_points[:, 1]), max(correl_points[:, 1])])
        zlim = tuple(float(x) for x in [min(correl_points[:, 2]), max(correl_points[:, 2])])
        points_3d_2 = Zscan.points3d_gpu(x_lim=xlim, y_lim=ylim, z_lim=zlim, z_step=.5, xy_step=2, visualize=False)
        print('X: {}x{}, Y: {}x{}, Z:{}x{}'.format(xlim[0], xlim[1], ylim[0], ylim[1], zlim[0], zlim[1]))
        print('2nd 3D meshgrid pts: {} mi'.format(points_3d_2.shape[0] / 1e6))
        print('Create second meshgrid pcl: {} s'.format(round(time.time() - t4, 2)))
        t4 = time.time()
        correl_points = Zscan.correlation_process(points_3d=points_3d_2, visualize=True, save_points=False,
                                                  win_size=3, threshold=0.90, method='temporal')
        print('Second Correl {}'.format(round(time.time() - t4, 2)))
        # median_z = np.median(correl_points[:, 2], axis=0)
        # output = (median_z - 1 < correl_points[:, 2]) & (correl_points[:, 2] < median_z + 1)
        # correl_points = correl_points[output]

        print('Full time: {} s'.format(round(time.time() - t0, 2)))
        Zscan.plot_3d_points(correl_points[:, 0], correl_points[:, 1], correl_points[:, 2],
                             title="Temporal x Spatial correlation result")
    #
    # Zscan.save_points(correl_points, filename='./sm4_parede_win7.csv')


if __name__ == "__main__":
    main()
