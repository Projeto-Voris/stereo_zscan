import os
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates

import rectify_matrix
import project_points
import cupy as cp



def plot_2d_planes(xyz):
    # Extract X, Y, Z coordinates
    # Extract X, Y, Z coordinates
    X = xyz[:, 0]
    Y = xyz[:, 1]
    Z = xyz[:, 2]

    # Create a figure with 2 rows and 3 columns for subplots
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))

    # Plot XY projection
    ax[0, 0].scatter(X, Y, c='b', marker='o')
    ax[0, 0].set_xlabel('X')
    ax[0, 0].set_ylabel('Y')
    ax[0, 0].set_title('XY Projection')
    ax[0, 0].grid()

    # Plot XZ projection
    ax[0, 1].scatter(X, Z, c='r', marker='o')
    ax[0, 1].set_xlabel('X')
    ax[0, 1].set_ylabel('Z')
    ax[0, 1].set_title('XZ Projection')
    ax[0, 1].grid()

    # Plot YZ projection
    ax[0, 2].scatter(Y, Z, c='g', marker='o')
    ax[0, 2].set_xlabel('Y')
    ax[0, 2].set_ylabel('Z')
    ax[0, 2].set_title('YZ Projection')
    ax[0, 2].grid()

    # Plot X distribution
    ax[1, 0].hist(X, bins=20, color='b', alpha=0.7)
    ax[1, 0].set_xlabel('X')
    ax[1, 0].set_ylabel('Frequency')
    ax[1, 0].set_title('X Distribution')

    # Plot Y distribution
    ax[1, 1].hist(Y, bins=20, color='r', alpha=0.7)
    ax[1, 1].set_xlabel('Y')
    ax[1, 1].set_ylabel('Frequency')
    ax[1, 1].set_title('Y Distribution')

    # Plot Z distribution
    ax[1, 2].hist(Z, bins=20, color='g', alpha=0.7)
    ax[1, 2].set_xlabel('Z')
    ax[1, 2].set_ylabel('Frequency')
    ax[1, 2].set_title('Z Distribution')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

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
        inter_Igray[:, n] = map_coordinates(images[:, :, n], projected_points_uv, order=1, mode='constant', cval=0)

    return inter_Igray


def temp_cross_correlation_gpu(left_Igray, right_Igray, points_3d):
    # Convert data to GPU arrays using CuPy
    left_Igray = cp.asarray(left_Igray, dtype=cp.float32)
    right_Igray = cp.asarray(right_Igray, dtype=cp.float32)

    # Calculate the mean of non-zero elements
    left_mean_inds = mean_ignore_zeros_gpu(left_Igray, axis=0)
    right_mean_inds = mean_ignore_zeros_gpu(right_Igray, axis=0)

    # Perform batch processing to avoid large memory allocations
    batch_size = 1000  # Adjust batch size to balance performance and memory usage
    num_points = left_Igray.shape[0]
    num_batches = (num_points + batch_size - 1) // batch_size  # Calculate number of batches

    hmax_list = []
    Imax_list = []

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_points)

        # Compute num and den for each batch using CuPy operations
        num = cp.sum(
            (left_Igray[start_idx:end_idx] - left_mean_inds) * (right_Igray[start_idx:end_idx] - right_mean_inds),
            axis=1)
        den = cp.sqrt(cp.sum((left_Igray[start_idx:end_idx] - left_mean_inds) ** 2, axis=1)) * cp.sqrt(
            cp.sum((right_Igray[start_idx:end_idx] - right_mean_inds) ** 2, axis=1))

        ho = num / den
        z_size = cp.unique(points_3d[:, 2]).shape[0]
        hmax = cp.zeros((end_idx - start_idx) // z_size, dtype=cp.float32)
        Imax = cp.zeros((end_idx - start_idx) // z_size, dtype=cp.float32)

        for k in range((end_idx - start_idx) // z_size):
            hmax[k] = cp.max(ho[k * z_size:(k + 1) * z_size])
            Imax[k] = cp.argmax(ho[k * z_size:(k + 1) * z_size]) + k * z_size + 1

        hmax_list.extend(cp.asnumpy(hmax))  # Convert to NumPy arrays if needed
        Imax_list.extend(cp.asnumpy(Imax))

    return cp.array(hmax_list), cp.array(Imax_list)


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

    return hmax, Imax


def main():
    # Paths for yaml file and images
    yaml_file = 'cfg/20240828_bouget.yaml'
    images_path = 'images/SM3-20240828 - GC (f50)'
    t0 = time.time()
    print('Initiate Algorithm')
    # Identify all images from path file
    left_images = project_points.read_images(os.path.join(images_path, 'left', ),
                                             sorted(os.listdir(os.path.join(images_path, 'left'))))

    right_images = project_points.read_images(os.path.join(images_path, 'right', ),
                                              sorted(os.listdir(os.path.join(images_path, 'right'))))
    t1 = time.time()
    print('Got {} left and right images: \n time: {} ms'.format(left_images.shape[2], round((t1 - t0) * 1e3, 2)))
    # Read file containing all calibration parameters from stereo system
    Kl, Dl, Rl, Tl, Kr, Dr, Rr, Tr, R, T = rectify_matrix.load_camera_params(yaml_file=yaml_file)

    t2 = time.time()
    print("Read yaml \n time: {} ms".format(round((t2 - t1) * 1e3, 2)))
    # Construct a 3D points (X,Y,Z) based on initial conditions and steps
    points_3d = project_points.points3d_cube(x_lim=(-100, 100), y_lim=(-100, 100), z_lim=(0, 5), xy_step=2, z_step=0.1,
                                             visualize=False)
    # Project points on Left and right
    uv_points_l = project_points.gcs2ccs(points_3d, Kl, Dl, Rl, Tl)
    uv_points_r = project_points.gcs2ccs(points_3d, Kr, Dr, Rr, Tr)
    _, valid_mask_l = project_points.filter_points_in_bounds(uv_points_l, left_images.shape[0], left_images.shape[1])
    _, valid_mask_r = project_points.filter_points_in_bounds(uv_points_r, right_images.shape[0], right_images.shape[1])
    # Filter values
    valid_mask = valid_mask_l & valid_mask_r
    t3 = time.time()
    print("Project points \n time: {} ms".format(round((t3 - t2) * 1e3, 2)))

    inter_points_L = interpolate_points(left_images, uv_points_l, method='linear')
    inter_points_R = interpolate_points(right_images, uv_points_r, method='linear')

    t4 = time.time()
    mask_colored = points_3d[:, 2] == 0
    xyz_colored = points_3d[mask_colored]
    # project_points.plot_3d_points(x=xyz_colored[:, 0], y=xyz_colored[:, 1], z=xyz_colored[:, 2],
    #                               color=inter_points_L[mask_colored, 0])
    # project_points.plot_3d_points(x=xyz_colored[:, 0], y=xyz_colored[:, 1], z=xyz_colored[:, 2],
    #                               color=inter_points_R[mask_colored, 0])

    print("Interpolate \n time: {} ms".format(round((t4 - t3) * 1e3, 2)))

    hmax, imax = temp_cross_correlation(inter_points_L, inter_points_R, points_3d)
    filtered_3d = points_3d[np.asarray(imax[hmax > 0.8], np.int32)]



    t5 = time.time()
    print("Cross Correlation yaml \n time: {} ms".format(round((t5 - t4) * 1e3, 2)))

    plt.figure()
    plt.plot(hmax)
    project_points.plot_3d_correl(filtered_3d[:, 0], filtered_3d[:, 1], filtered_3d[:, 2], correl=hmax[hmax > 0.8])

    t6 = time.time()
    print('Ploted 3D points and correlation data \n time: {} ms'.format(round((t6 - t5) * 1e3, 2)))
    print("Points max correlation: {}".format(filtered_3d.shape[0]))
    plot_2d_planes(filtered_3d)


if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
