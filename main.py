import numpy as np
import matplotlib.pyplot as plt
import os
import time
from include.InverseTriangulation import InverseTriangulation
from extras.debugger import load_array_from_csv
from extras.project_points import read_images, points3d_cube


def main():
    yaml_file = 'cfg/SM4_20241018_bouget.yaml'
    images_path = 'images/SM4-20241112 - close'

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
    t1 = time.time()
    print('Open Correlation images: {} s'.format(round(t1 - t0, 2)))
    # points_3d = points3d_cube(x_lim=(-50, 50), y_lim=(50, 50), z_lim=(-200, 200),
    #                           xy_step=5, z_step=0.1, visualize=False)  # pontos para SM4
    #
    # print("Create meshgrid of {} points: {} s".format(points_3d.shape[0], round(time.time() - t1, 2)))

    n_imgs_v = [10]
    for n_img in n_imgs_v:
        t2 = time.time()
        Zscan = InverseTriangulation(yaml_file=yaml_file)
        Zscan.read_images(right_imgs=right_images[:, :, :n_img], left_imgs=left_images[:, :, :n_img])
        points_3d = Zscan.points3d(x_lim=(-0, 350), y_lim=(-100, 400), z_lim=(100, 300), xy_step=5, z_step=0.1,
                                   visualize=False)
        # points_3d = Zscan.points3d(x_lim=(-0, 10), y_lim=(0, 10), z_lim=(0, 10), xy_step=2, z_step=5,
        #                            visualize=False)
        print('Create mesgrid pcl: {} s'.format(round(time.time() - t2, 2)))
        t3 = time.time()
        # z_lin = np.split(np.arange(-1000, 1000, 0.1), 10)
        # for zlin in z_lin:
        #     points_3d = Zscan.points3d_zstep(x_lim=(-0, 350), y_lim=(-100, 400), z_lin=zlin, xy_step=2, visualize=False)
        correl_points = Zscan.correlation_process(points_3d=points_3d, visualize=True, save_points=False, win_size=7)

    print('Full time: {} s'.format(round(time.time() - t0, 2)))

    # Inverse Triangulation for correlation
    # Zscan.read_images(left_imgs=left_images, right_imgs=right_images)
    # Zscan.correlation_process(win_size=5, correl_param=(0.3, 0.7), save_points=False, visualize=True)

    # Inverse Triangulation for fringe projection
    # Zscan.read_images(left_imgs=left_image, right_imgs=right_image)
    # z_zcan_points = Zscan.fringe_process(save_points=True, visualize=True)


if __name__ == "__main__":
    main()
