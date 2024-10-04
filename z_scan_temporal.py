import os
import time
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.stats import rankdata, norm, spearmanr

import debugger
import rectify_matrix
import project_points


def interpolate_points(images, projected_points):
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
        inter_Igray[:, n] = map_coordinates(images[:, :, n], projected_points_uv, order=1, mode='constant', cval=0)

    return inter_Igray


def temp_cross_correlation(left_Igray, right_Igray, points_3d):
    # Mean values along time
    left_mean_Igray = np.mean(left_Igray, axis=1)
    right_mean_Igray = np.mean(right_Igray, axis=1)
    ho_ztep = []
    # Correlation equation from https://linkinghub.elsevier.com/retrieve/pii/S0143816612000759
    num = np.sum((left_Igray - left_mean_Igray[:, None]) * (right_Igray - right_mean_Igray[:, None]), axis=1)
    den = np.sqrt(np.sum((left_Igray - left_mean_Igray[:, None]) ** 2, axis=1) * np.sum(
        (right_Igray - right_mean_Igray[:, None]) ** 2, axis=1))
    ho = num / np.maximum(den, 1e-10)

    # Number of tested Z
    z_size = np.unique(points_3d[:, 2]).shape[0]

    # Construct vector to save hmax and Id max
    hmax = np.zeros(int(points_3d.shape[0] / z_size))
    hmin = np.zeros(int(points_3d.shape[0] / z_size))
    Imax = np.zeros(int(points_3d.shape[0] / z_size))
    Imin = np.zeros(int(points_3d.shape[0] / z_size))

    for k in range(points_3d.shape[0] // z_size):
        # Extract the range of correlation values corresponding to one (X, Y) over all Z values
        ho_range = ho[k * z_size:(k + 1) * z_size]
        ho_ztep.append(ho_range)
        # Find the maximum correlation for this (X, Y) pair
        hmax[k] = np.nanmax(ho_range)
        hmin[k] = np.nanmin(ho_range)
        # Find the index of the maximum correlation value
        Imax[k] = np.nanargmax(ho_range) + k * z_size
        Imin[k] = np.nanargmin(ho_range) + k * z_size

    return ho, hmax, Imax, ho_ztep, hmin, Imin


def pearson_correlation(left_Igray, right_Igray, points_3d):
    pearson_correl = []
    z_size = np.unique(points_3d[:, 2]).shape[0]
    pearson_max = np.zeros(int(points_3d.shape[0] / z_size))
    id_max = np.zeros(int(points_3d.shape[0] / z_size))
    pearson_zstep = []
    for k in range(points_3d.shape[0] // z_size):
        pearson_correl_range = np.asarray(
            [np.corrcoef(left_Igray[i, :], right_Igray[i, :])[0, 1] for i in range(k * z_size, (k + 1) * z_size)],
            np.float32)
        pearson_max[k] = np.nanmax(pearson_correl_range)
        id_max[k] = np.nanargmax(pearson_correl_range) + k * z_size
        np.concatenate((pearson_correl, pearson_correl_range))
        pearson_zstep.append(pearson_correl_range)
    return pearson_correl, pearson_max, id_max, pearson_zstep


def spearman_correlation(left_Igray, right_Igray, points_3d):
    spearman_correl = []
    z_size = np.unique(points_3d[:, 2]).shape[0]
    spearman_max = np.zeros(int(points_3d.shape[0] / z_size))
    id_max = np.zeros(int(points_3d.shape[0] / z_size))
    spearman_zstep = []
    for k in range(points_3d.shape[0] // z_size):
        spearman_correl_range = np.asarray(
            [spearmanr(left_Igray[i, :], right_Igray[i, :]) for i in range(k * z_size, (k + 1) * z_size)],
            np.float32)[:, 0]
        spearman_max[k] = np.nanmax(spearman_correl_range)
        id_max[k] = np.nanargmax(spearman_correl_range) + k * z_size
        np.concatenate((spearman_correl, spearman_correl_range))
        spearman_zstep.append(spearman_correl_range)
    return spearman_correl, spearman_max, id_max, spearman_zstep


def main():
    # Paths for yaml file and images
    yaml_file = 'cfg/SM4_20241004_bouget.yaml'
    # images_path = 'images/SM3-20240918 - noise'
    images_path = 'images/SM4-20241004 - noise'
    Nimg = 40
    DEBUG = True
    t0 = time.time()
    print('Initiate Algorithm with {} images'.format(Nimg))
    # Identify all images from path file
    left_images = project_points.read_images(os.path.join(images_path, 'left', ),
                                             sorted(os.listdir(os.path.join(images_path, 'left'))),
                                             n_images=Nimg, visualize=False)

    right_images = project_points.read_images(os.path.join(images_path, 'right', ),
                                              sorted(os.listdir(os.path.join(images_path, 'right'))),
                                              n_images=Nimg, visualize=False)

    t1 = time.time()
    print('Got {} left and right images: \n dt: {} ms'.format(right_images.shape[2] + left_images.shape[2],
                                                              round((t1 - t0) * 1e3, 2)))

    # Read file containing all calibration parameters from stereo system
    Kl, Dl, Rl, Tl, Kr, Dr, Rr, Tr, R, T = rectify_matrix.load_camera_params(yaml_file=yaml_file)

    # Construct a 3D points (X,Y,Z) based on initial conditions and steps
    # points_3d = project_points.points3d_cube(x_lim=(0, 155), y_lim=(0, 105), z_lim=(-100, 100), xy_step=50, z_step=0.1,
    #                                          visualize=False)
    points_3d = project_points.points3d_cube(x_lim=(0, 50), y_lim=(0, 50), z_lim=(-100, 100), xy_step=50, z_step=0.1,
                                             visualize=False)
    print(points_3d[:10,:])
    # Project points on Left and right
    uv_points_l = project_points.gcs2ccs(points_3d, Kl, Dl, Rl, Tl)
    uv_points_r = project_points.gcs2ccs(points_3d, Kr, Dr, Rr, Tr)

    k = 0
    if DEBUG:
        reproj_l = debugger.plot_points_on_image(left_images[:, :, k], uv_points_l)
        reproj_r = debugger.plot_points_on_image(right_images[:, :, k], uv_points_r)
        crop_l = debugger.crop_img2proj_points(reproj_l, uv_points_l)
        crop_r = debugger.crop_img2proj_points(reproj_r, uv_points_r)
        cv2.imshow('croped r', crop_r)
        cv2.imshow('croped l', crop_l)
        debugger.show_stereo_images(reproj_l, reproj_r, 'Reprojected points {} image'.format(k))
        cv2.destroyAllWindows()
    t3 = time.time()
    print("Project points \n dt: {} ms".format(round((t3 - t1) * 1e3, 2)))

    # Interpolate reprojected points to image bounds (return pixel intensity)
    inter_points_L = interpolate_points(left_images, uv_points_l)
    inter_points_R = interpolate_points(right_images, uv_points_r)

    t4 = time.time()

    print("Interpolate \n dt: {} ms".format(round((t4 - t3) * 1e3, 2)))

    # Temporal correlation for L and R interpolated points
    ho, hmax, imax, ho_zstep, hmin, imin = temp_cross_correlation(inter_points_L, inter_points_R, points_3d)
    filtered_3d_ho = points_3d[np.asarray(imax, np.int32)]
    filtered_3d_ho_min = points_3d[np.asarray(imin, np.int32)]
    # spearman_correl, sp_max, id_s_max, sp_zstep = spearman_correlation(inter_points_L, inter_points_R,  points_3d)
    # filtered_3d_spearman = points_3d[np.asarray(id_s_max, np.int32)]
    # p_correl, p_max, id_p_max, pearson_zstep = pearson_correlation(inter_points_L, inter_points_R, points_3d)
    # filtered_3d_perason = points_3d[np.asarray(id_p_max, np.int32)]
    t5 = time.time()
    print("Cross Correlation yaml \n dt: {} ms".format(round((t5 - t4) * 1e3, 2)))

    # debugger.plot_point_correl(points_3d, ho)
    # debugger.plot_point_correl(points_3d, p_correl)
    debugger.plot_3d_points(filtered_3d_ho[:, 0], filtered_3d_ho[:, 1], filtered_3d_ho[:, 2], color=hmax,
                            title="Article Correl total for {} images".format(Nimg))
    debugger.plot_3d_points(filtered_3d_ho_min[:, 0], filtered_3d_ho_min[:, 1], filtered_3d_ho_min[:, 2], color=hmax,
                            title="Article Correl total for {} images".format(Nimg))
    # debugger.plot_3d_points(filtered_3d_perason[:, 0], filtered_3d_perason[:, 1], filtered_3d_perason[:, 2],
    #                         color=p_max, title="Pearson Correl for {} images".format(Nimg))
    # debugger.plot_3d_correl(filtered_3d_spearman[:, 0], filtered_3d_spearman[:, 1], filtered_3d_spearman[:, 2],
    #                         correl=hmax, title="Spearmanr Correl for {} images".format(Nimg))
    t6 = time.time()
    print('Ploted 3D points and correlation data \n dt: {} ms'.format(round((t6 - t5) * 1e3, 2)))
    # print("Points max correlation: {}".format(filtered_3d_ho.shape[0]))
    plt.figure()
    plt.plot(ho)
    plt.show()
    print(np.argmax(ho), ' , ', np.max(ho))

    img_l = cv2.circle(cv2.cvtColor(np.uint8(left_images[:, :, 0]), cv2.COLOR_GRAY2BGR),
                       (int(uv_points_l[0, np.argmax(ho)]), int(uv_points_l[1, np.argmax(ho)])), 5, (0, 255, 0), 2)
    img_r = cv2.circle(cv2.cvtColor(np.uint8(right_images[:, :, 0]), cv2.COLOR_GRAY2BGR),
                       (int(uv_points_r[0, np.argmax(ho)]), int(uv_points_r[1, np.argmax(ho)])), 5, (0, 255, 0), 2)
    # debugger.show_stereo_images(img_l, img_r)

    cv2.namedWindow('left', cv2.WINDOW_NORMAL)
    cv2.namedWindow('right', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('left', 1920, 1080)
    cv2.resizeWindow('right', 1920, 1080)
    cv2.imshow('left', img_l)
    cv2.imshow('right', img_r)
    cv2.waitKey(0)
    print('wait')

    # plot_2d_planes(filtered_3d)


if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
