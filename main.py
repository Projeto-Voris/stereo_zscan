import numpy as np
import matplotlib.pyplot as plt
import os
import time
from include.InverseTriangulation import InverseTriangulation
from extras.debugger import load_array_from_csv
from extras.project_points import read_images, points3d_cube

def main():
    yaml_file = 'cfg/SM4_20241004_bianca.yaml'
    images_path = 'images/SM4-20241004 - noise'

    Nimg = 10
    fringe_image_name = '016.csv'
    t0 = time.time()
    left_images = read_images(os.path.join(images_path, 'left', ),
                              sorted(os.listdir(os.path.join(images_path, 'left'))),
                              n_images=Nimg, visualize=False)

    right_images = read_images(os.path.join(images_path, 'right', ),
                               sorted(os.listdir(os.path.join(images_path, 'right'))),
                               n_images=Nimg, visualize=False)
    left_image = load_array_from_csv('csv/left/left_abs_{}'.format(fringe_image_name))
    right_image = load_array_from_csv('csv/right/right_abs_{}'.format(fringe_image_name))
    t1 = time.time()
    print('Open Correlation images: {} s'.format(round(t1 - t0, 2)))
    points_3d = points3d_cube(x_lim=(-50, 250), y_lim=(-150, 200), z_lim=(-100, 200),
                              xy_step=5, z_step=0.1, visualize=False)  # pontos para SM4

    print("Create meshgrid of {} points: {} s".format(points_3d.shape[0], round(time.time() - t1, 2)))

    Zscan = InverseTriangulation(yaml_file=yaml_file, gcs3d_pts=points_3d)

    # Inverse Triangulation for correlation
    Zscan.read_images(left_imgs=left_images, right_imgs=right_images)
    Zscan.correlation_process(win_size=5, correl_param=(0.3, 0.7), save_points=False, visualize=True)


    # Inverse Triangulation for fringe projection
    Zscan.read_images(left_imgs=left_image, right_imgs=right_image)
    z_zcan_points = Zscan.fringe_process(save_points=False, visualize=True)

if __name__ == "__main__":
    main()
