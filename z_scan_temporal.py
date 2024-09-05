import os
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates

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

    inter_ids = np.zeros((projected_points_uv.shape[1], images.shape[2]), dtype=float)

    for n in range(images.shape[2]):
        # Create the interpolation function for each image

        # Perform interpolation for all projected points
        inter_ids[:, n] = map_coordinates(images[:, :, n], projected_points_uv, order=1, mode='constant', cval=0)

    return inter_ids


def mean_ignore_zeros_gpu(arr, axis=None):
    # Create a mask of non-zero elements
    non_zero_mask = arr != 0

    # Count non-zero elements along the specified axis
    non_zero_count = cp.sum(non_zero_mask, axis=axis)

    # Sum the non-zero elements along the specified axis
    non_zero_sum = cp.sum(arr * non_zero_mask, axis=axis)

    # Calculate the mean by dividing the sum by the count
    mean = cp.divide(non_zero_sum, non_zero_count, where=non_zero_count != 0)  # Avoid division by zero

    return mean


def temp_cross_correlation_gpu(left_ids, right_ids, points_3d):
    # Convert data to GPU arrays using CuPy
    left_ids = cp.asarray(left_ids, dtype=cp.float32)
    right_ids = cp.asarray(right_ids, dtype=cp.float32)

    # Calculate the mean of non-zero elements
    left_mean_inds = mean_ignore_zeros_gpu(left_ids, axis=0)
    right_mean_inds = mean_ignore_zeros_gpu(right_ids, axis=0)

    # Perform batch processing to avoid large memory allocations
    batch_size = 1000  # Adjust batch size to balance performance and memory usage
    num_points = left_ids.shape[0]
    num_batches = (num_points + batch_size - 1) // batch_size  # Calculate number of batches

    hmax_list = []
    Imax_list = []

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_points)

        # Compute num and den for each batch using CuPy operations
        num = cp.sum(
            (left_ids[start_idx:end_idx] - left_mean_inds) * (right_ids[start_idx:end_idx] - right_mean_inds), axis=1)
        den = cp.sqrt(cp.sum((left_ids[start_idx:end_idx] - left_mean_inds) ** 2, axis=1)) * cp.sqrt(
            cp.sum((right_ids[start_idx:end_idx] - right_mean_inds) ** 2, axis=1))

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


def mean_ignore_zeros(arr, axis=None):
    # Create a mask of non-zero elements
    non_zero_mask = arr != 0

    # Count non-zero elements along the specified axis
    non_zero_count = np.sum(non_zero_mask, axis=axis)

    # Sum the non-zero elements along the specified axis
    non_zero_sum = np.sum(arr * non_zero_mask, axis=axis)

    # Calculate the mean by dividing the sum by the count
    mean = np.divide(non_zero_sum, non_zero_count, where=non_zero_count != 0)  # Avoid division by zero

    return mean


def temp_cross_correlation(left_ids, right_ids, points_3d):
    # left_mean_ids = mean_ignore_zeros(left_ids, axis=0)
    # right_mean_ids = mean_ignore_zeros(right_ids, axis=0)

    left_mean_ids = np.mean(left_ids, axis=0)
    right_mean_ids = np.mean(right_ids, axis=0)
    num = np.sum((left_ids - left_mean_ids) * (right_ids - right_mean_ids), axis=1)
    den = np.sqrt(np.sum((left_ids - left_mean_ids) ** 2, axis=1)) * np.sqrt(
        np.sum((right_ids - right_mean_ids) ** 2, axis=1))
    ho = num / den

    z_size = np.unique(points_3d[:, 2]).shape[0]
    hmax = np.zeros(int(points_3d.shape[0] / z_size))
    Imax = np.zeros(int(points_3d.shape[0] / z_size))

    for k in range(points_3d.shape[0] // z_size):
        hmax[k] = np.max(ho[k * z_size:(k + 1) * z_size])
        Imax[k] = np.argmax(ho[k * z_size:(k + 1) * z_size])  + k * z_size

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
    t1 = round(time.time() - t0, 2)
    print('Got {} left and right images: \n time: {} ms'.format(left_images.shape[2], t1))
    # Read file containing all calibration parameters from stereo system
    Kl, Dl, Rl, Tl, Kr, Dr, Rr, Tr, _, _ = rectify_matrix.load_camera_params(yaml_file=yaml_file)

    t2 = round(time.time() - t1, 2)
    print("Read yaml \n time: {} ms".format(time.time() - t2))
    # Construct a 3D points (X,Y,Z) based on initial conditions and steps
    points_3d = project_points.points3d_cube(x_lim=(-100, 100), y_lim=(-100, 100), z_lim=(0, 10), xy_step=1, z_step=0.1,
                                             visualize=False)
    uv_points_l = project_points.gcs2ccs(points_3d, Kl, Dl, Rl, Tl)
    uv_points_r = project_points.gcs2ccs(points_3d, Kr, Dr, Rr, Tr)
    t3 = round(time.time() - t2, 2)
    print("Project points \n time: {} ms".format(time.time() - t3))
    inter_points_L = interpolate_points(left_images, uv_points_l, method='linear')
    inter_points_R = interpolate_points(right_images, uv_points_r, method='linear')
    t4 = round(time.time() - t3, 2)
    print("Interpolate \n time: {} ms".format(time.time() - t4))
    hmax, imax = temp_cross_correlation(inter_points_L, inter_points_R, points_3d)
    filtered_3d = points_3d[np.array(imax, dtype=int)]
    t5 = round(time.time() - t4, 2)
    print("Cross Correlation yaml \n time: {} ms".format(time.time() - t5))
    plt.figure()
    plt.plot(hmax)
    project_points.plot_3d_points(filtered_3d[:, 0], filtered_3d[:, 1], filtered_3d[:, 2])
    print('wait')
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
