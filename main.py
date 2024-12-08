import gc

import numpy as np
import os
import time

from include.InverseTriangulation import InverseTriangulation
from extras.project_points import read_images

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
        points_3d = Zscan.points3d_gpu(x_lim=(-600, 600), y_lim=(-600, 600), z_lim=(-800, 800), xy_step=15, z_step=2,
                                   visualize=False)
        print('3D meshgrid pts: {} mi '.format(points_3d.shape[0]/1e6))
        print('Create mesgrid pcl: {} s'.format(round(time.time() - t2, 2)))
        t3 = time.time()

        correl_points = Zscan.correlation_process(points_3d=points_3d, visualize=True, save_points=False,
                                                  win_size=7, threshold=0.9)
        print('First Correl {}'.format(round(time.time() - t3, 2)))
        t4 = time.time()

        xlim, ylim, zlim = (min(correl_points[:,0]), max(correl_points[:,0])), (min(correl_points[:,1]), max(correl_points[:,1])), (min(correl_points[:,2]), max(correl_points[:,2]))
        points_3d_2 = Zscan.points3d_gpu(x_lim=xlim, y_lim=ylim, z_lim=zlim, z_step=.1, xy_step=2, visualize=False)
        print('X: {}x{}, Y: {}x{}, Z:{}x{}'.format(xlim[0], xlim[1], ylim[0], ylim[1], zlim[0], zlim[1]))
        print('2nd 3D meshgrid pts: {} mi'.format(points_3d_2.shape[0]/1e6))
        print('Create second meshgrid pcl: {} s'.format(round(time.time() - t4, 2)))
        t4 = time.time()
        correl_points = Zscan.correlation_process(points_3d=points_3d_2, visualize=True, save_points=False,
                                                  win_size=5, threshold=0.8)

        print('Second Correl {}'.format(round(time.time() - t4, 2)))
        # Generate region bounds for x and y
        x_step = 100  # Define step size for x regions
        y_step = 100  # Define step size for y regions

        x_bins = np.arange(min(correl_points[:, 0]), max(correl_points[:, 0]) + x_step, x_step)
        y_bins = np.arange(min(correl_points[:, 1]), max(correl_points[:, 1]) + y_step, y_step)

        x_regions = [(x_bins[i], x_bins[i + 1]) for i in range(len(x_bins) - 1)]
        y_regions = [(y_bins[i], y_bins[i + 1]) for i in range(len(y_bins) - 1)]

        # Create a list of (x, y) region pairs
        xy_regions = [(x_range, y_range) for x_range in x_regions for y_range in y_regions]
        # Split points into regions based on X and Y ranges
        region_points = {}
        # for (x_range, y_range) in xy_regions:
        #     x_low, x_high = x_range
        #     y_low, y_high = y_range
        #
        #     # Filter points within the current x-y region
        #     points_in_region = correl_points[
        #         (correl_points[:, 0] >= x_low) & (correl_points[:, 0] < x_high) &
        #         (correl_points[:, 1] >= y_low) & (correl_points[:, 1] < y_high)
        #         ]
        #     region_key = f"x({x_low}-{x_high})_y({y_low}-{y_high})"
        #     region_points[region_key] = points_in_region
        # Process regions in parallel
        # output_3d = []
        # for i, split in region_points.items():
        #     if split.shape[0] == 0:
        #         print(f"Skipping empty region: {i}")
        #         continue
        #     xlim = [min(split[:, 0]), max(split[:, 0])]
        #     ylim = [min(split[:, 1]), max(split[:, 1])]
        #     zlim = [min(split[:, 2]), max(split[:, 2])]
        #     print('{} - X: {}x{}, Y: {}x{}, Z:{}x{}'.format(i, xlim[0], xlim[1], ylim[0], ylim[1], zlim[0], zlim[1]))
        #     print('{} - 3D meshgrid pts: {} mi'.format(i, points_3d_2.shape[0] / 1e6))
        #     points_3d = Zscan.points3d(x_lim=xlim, y_lim=ylim, z_lim=zlim, z_step=0.05, xy_step=1, visualize=False)
        #     correl_points = Zscan.correlation_process(points_3d=points_3d, visualize=True, save_points=False,
        #                                               win_size=7, threshold=0.8)
        #     output_3d.append(correl_points)
        #
        #     # Free memory after processing each region
        #     del points_3d, correl_points
        #     gc.collect()

    print('Full time: {} s'.format(round(time.time() - t0, 2)))

    # Zscan.save_points(correl_points, filename='./sm4_parede_win7.csv')



if __name__ == "__main__":
    main()
