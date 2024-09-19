import os
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from scipy.ndimage import maximum_filter

import debugger
import rectify_matrix
import project_points
import cupy as cp





def interpolate_points(images, projected_points, method='linear'):
    """
    Interpolate points from images and projected points.

    Parameters:
    images: (height, width, num_images) array of images.
    projected_points: (3, N) array of projected points in homogeneous coordinates (u, v, 1).
    method: (string) interpolation method.

    Returns:
    inter_pts: (N, num_images) array of interpolated points.
    """
    # Remove the homogeneous coordinate and transpose to (N, 2) format
    projected_points_uv = projected_points[:2, :]  # Shape: (2,N)

    inter_Igray = np.zeros((projected_points_uv.shape[1], images.shape[2]), dtype=float)

    for n in range(images.shape[2]):
        # Create the interpolation function for each image

        # Perform interpolation for all projected points
        inter_Igray[:, n] = map_coordinates(images[:, :, n], projected_points_uv, order=3, mode='constant', cval=0)

    return inter_Igray


def temp_cross_correlation(left_Igray, right_Igray, points_3d):
    # Mean values along time
    left_mean_Igray = np.mean(left_Igray, axis=1)
    right_mean_Igray = np.mean(right_Igray, axis=1)
    # Correlation equation from https://linkinghub.elsevier.com/retrieve/pii/S0143816612000759
    num = np.sum((left_Igray - left_mean_Igray[:, None]) * (right_Igray - right_mean_Igray[:, None]), axis=1)
    den = np.sqrt(np.sum((left_Igray - left_mean_Igray[:, None]) ** 2, axis=1)) * np.sqrt(
        np.sum((right_Igray - right_mean_Igray[:, None]) ** 2, axis=1))
    ho = num / np.maximum(den, 1e-10)

    # Number of tested Z
    z_size = np.unique(points_3d[:, 2]).shape[0]

    # Construct vector to save hmax and Id max
    hmax = np.zeros(int(points_3d.shape[0] / z_size))
    Imax = np.zeros(int(points_3d.shape[0] / z_size))

    for k in range(points_3d.shape[0] // z_size):
        # Extract the range of correlation values corresponding to one (X, Y) over all Z values
        ho_range = ho[k * z_size:(k + 1) * z_size]

        # Find the maximum correlation for this (X, Y) pair
        hmax[k] = np.nanmax(ho_range)

        # Find the index of the maximum correlation value
        Imax[k] = np.nanargmax(ho_range) + k * z_size

    return hmax, Imax, ho
def main():
    # Paths for yaml file and images
    yaml_file = 'cfg/20240918_bouget.yaml'
    images_path = '/home/daniel/Insync/daniel.regner@labmetro.ufsc.br/Google Drive - Shared drives/VORIS  - Equipe/Sistema de Medição 3 - Stereo Ativo - Projeção Laser/Imagens/Testes/SM3-20240919 - noise 4.0'
    Nimg = 20
    t0 = time.time()
    print('Initiate Algorithm with {} images'.format(Nimg))
    # Identify all images from path file
    left_images = project_points.read_images(os.path.join(images_path, 'left', ),
                                             sorted(os.listdir(os.path.join(images_path, 'left'))), n_images=Nimg)

    right_images = project_points.read_images(os.path.join(images_path, 'right', ),
                                              sorted(os.listdir(os.path.join(images_path, 'right'))), n_images=Nimg)

    t1 = time.time()
    print('Got {} left and right images: \n dt: {} ms'.format(right_images.shape[2]+left_images.shape[2], round((t1 - t0) * 1e3, 2)))

    # Read file containing all calibration parameters from stereo system
    Kl, Dl, Rl, Tl, Kr, Dr, Rr, Tr, R, T = rectify_matrix.load_camera_params(yaml_file=yaml_file)


    # Construct a 3D points (X,Y,Z) based on initial conditions and steps
    points_3d = project_points.points3d_cube(x_lim=(80, 100), y_lim=(80, 100), z_lim=(-500, 500), xy_step=1, z_step=0.1,
                                             visualize=False)
    # Project points on Left and right
    uv_points_l = project_points.gcs2ccs(points_3d, Kl, Dl, Rl, Tl)
    uv_points_r = project_points.gcs2ccs(points_3d, Kr, Dr, Rr, Tr)
    # _, valid_mask_l = project_points.filter_points_in_bounds(uv_points_l, left_images.shape[0], left_images.shape[1])
    # _, valid_mask_r = project_points.filter_points_in_bounds(uv_points_r, right_images.shape[0], right_images.shape[1])
    # # Filter values
    # valid_mask = valid_mask_l & valid_mask_r

    t3 = time.time()
    print("Project points \n dt: {} ms".format(round((t3 - t1) * 1e3, 2)))

    # Interpolate reprojected points to image bounds (return pixel intensity)
    inter_points_L = interpolate_points(left_images, uv_points_l, method='linear')
    inter_points_R = interpolate_points(right_images, uv_points_r, method='linear')

    t4 = time.time()

    print("Interpolate \n dt: {} ms".format(round((t4 - t3) * 1e3, 2)))

    # Temporal correlation for L and R interpolated points
    hmax, imax, ho = temp_cross_correlation(inter_points_L, inter_points_R, points_3d)
    filtered_3d_max = points_3d[np.asarray(imax[hmax > 0.8], np.int32)]
    filtered_3d = points_3d[np.asarray(imax, np.int32)]

    t5 = time.time()
    print("Cross Correlation yaml \n dt: {} ms".format(round((t5 - t4) * 1e3, 2)))

    debugger.plot_point_correl(points_3d, ho)
    debugger.plot_3d_correl(filtered_3d[:, 0], filtered_3d[:, 1], filtered_3d[:, 2], correl=hmax, title="Hmax total for {} images".format(Nimg))
    t6 = time.time()
    print('Ploted 3D points and correlation data \n dt: {} ms'.format(round((t6 - t5) * 1e3, 2)))
    print("Points max correlation: {}".format(filtered_3d.shape[0]))
    # plot_2d_planes(filtered_3d)


if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
