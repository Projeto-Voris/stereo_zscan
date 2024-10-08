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
from scipy.interpolate import griddata

import numpy as np

import numpy as np

import numpy as np


def bi_interpolation(images, uv_points, batch_size=10000):
    """
    Perform bilinear interpolation on a stack of images at specified uv_points, optimized for memory.

    Parameters:
    images: (height, width, num_images) array of images, or (height, width) for a single image.
    uv_points: (2, N) array of points where N is the number of points.
    batch_size: Maximum number of points to process at once (default 10,000 for memory efficiency).

    Returns:
    interpolated: (N, num_images) array of interpolated pixel values, or (N,) for a single image.
    std: Standard deviation of the corner pixels used for interpolation.
    """
    if len(images.shape) == 2:
        # Convert 2D image to 3D for consistent processing
        images = images[:, :, np.newaxis]

    height, width, num_images = images.shape
    N = uv_points.shape[1]

    # Initialize the output arrays
    interpolated = np.zeros((N, num_images))
    std = np.zeros(N)

    # Process points in batches
    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        uv_batch = uv_points[:, i:end]

        x = uv_batch[0].astype(float)
        y = uv_batch[1].astype(float)

        # Ensure x and y are within bounds
        x1 = np.clip(np.floor(x).astype(int), 0, width - 1)
        y1 = np.clip(np.floor(y).astype(int), 0, height - 1)
        x2 = np.clip(x1 + 1, 0, width - 1)
        y2 = np.clip(y1 + 1, 0, height - 1)

        # Calculate the differences
        x_diff = x - x1
        y_diff = y - y1

        # Bilinear interpolation in batches (vectorized)
        for n in range(num_images):
            p11 = images[y1, x1, n]  # Top-left corner
            p12 = images[y2, x1, n]  # Bottom-left corner
            p21 = images[y1, x2, n]  # Top-right corner
            p22 = images[y2, x2, n]  # Bottom-right corner

            # Bilinear interpolation formula (for each batch)
            interpolated_batch = (
                    p11 * (1 - x_diff) * (1 - y_diff) +
                    p21 * x_diff * (1 - y_diff) +
                    p12 * (1 - x_diff) * y_diff +
                    p22 * x_diff * y_diff
            )
            interpolated[i:end, n] = interpolated_batch

            # # Compute standard deviation across the four corners for each point
            # std_batch = np.std(np.vstack([p11, p12, p21, p22]), axis=0)
            # std[i:end] = std_batch

    # Return 1D interpolated result if the input was a 2D image
    if images.shape[2] == 1:
        interpolated = interpolated[:, 0]
    std = np.zeros_like((uv_points.shape[0], images.shape[2]))
    return interpolated, std



def temp_cross_correlation(left_Igray, right_Igray, points_3d, batch_size=10000):
    """
    Calculate the cross-correlation between two sets of images over time in batches.

    Parameters:
    left_Igray: (num_images, num_points) array of left images in grayscale.
    right_Igray: (num_images, num_points) array of right images in grayscale.
    points_3d: (N, 3) array of 3D points.
    batch_size: Size of batches to process at once.

    Returns:
    ho: Cross-correlation values.
    hmax: Maximum correlation values.
    Imax: Indices of maximum correlation values.
    """
    num_points = left_Igray.shape[1]
    num_batches = (num_points + batch_size - 1) // batch_size  # Calculate number of batches

    # Initialize ho to store cross-correlation values
    ho = np.zeros(num_points)

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_points)

        # Compute means
        left_mean_Igray = np.mean(left_Igray[:, start_idx:end_idx], axis=1, keepdims=True)
        right_mean_Igray = np.mean(right_Igray[:, start_idx:end_idx], axis=1, keepdims=True)

        # Compute numerator and denominator in batch
        num = np.sum((left_Igray[:, start_idx:end_idx] - left_mean_Igray) *
                     (right_Igray[:, start_idx:end_idx] - right_mean_Igray), axis=1)

        left_sq_diff = np.sum((left_Igray[:, start_idx:end_idx] - left_mean_Igray) ** 2, axis=1)
        right_sq_diff = np.sum((right_Igray[:, start_idx:end_idx] - right_mean_Igray) ** 2, axis=1)

        den = np.sqrt(left_sq_diff * right_sq_diff)
        ho[start_idx:end_idx] = num / np.maximum(den, 1e-10)

    # Number of tested Z
    z_size = np.unique(points_3d[:, 2]).shape[0]
    num_pairs = num_points // z_size

    # Preallocate arrays for maximum correlation and their indices
    hmax = np.zeros(num_pairs)
    Imax = np.zeros(num_pairs, dtype=int)

    for k in range(num_pairs):
        # Extract the range of correlation values corresponding to one (X, Y) over all Z values
        ho_range = ho[k * z_size:(k + 1) * z_size]

        # Find the maximum correlation for this (X, Y) pair
        hmax[k] = np.nanmax(ho_range)
        # Find the index of the maximum correlation value
        Imax[k] = np.nanargmax(ho_range) + k * z_size

    return ho, hmax, Imax




def phase_map(left_Igray, right_Igray, points_3d):
    z_step = np.unique(points_3d[:, 2]).shape[0]
    phi_map = []
    phi_min = []
    phi_min_id = []
    for k in range(points_3d.shape[0] // z_step):
        diff_phi = np.abs(left_Igray[z_step * k:(k + 1) * z_step] - right_Igray[z_step * k:(k + 1) * z_step])
        phi_map.append(diff_phi)
        phi_min.append(np.nanmin(diff_phi))
        phi_min_id.append(np.argmin(diff_phi) + k * z_step)

    return phi_map, phi_min, phi_min_id


def fringe_masks(image, uv_l, uv_r, std_l, std_r, phi_id):
    valid_u_l = (uv_l[0, :] >= 0) & (uv_l[0, :] < image.shape[1])
    valid_v_l = (uv_l[1, :] >= 0) & (uv_l[1, :] < image.shape[0])
    valid_u_r = (uv_r[0, :] >= 0) & (uv_r[0, :] < image.shape[1])
    valid_v_r = (uv_r[1, :] >= 0) & (uv_r[1, :] < image.shape[0])
    valid_uv = valid_u_l & valid_u_r & valid_v_l & valid_v_r
    phi_mask = np.zeros(uv_l.shape[1], dtype=bool)
    phi_mask[phi_id] = True
    thresh = 0.5
    valid_std = (std_l < thresh) & (std_r < thresh)

    return valid_uv & valid_std & phi_mask


def correl_mask(image, uv_l, uv_r, std_l, std_r, hmax):
    valid_u_l = (uv_l[0, :] >= 0) & (uv_l[0, :] < image.shape[1])
    valid_v_l = (uv_l[1, :] >= 0) & (uv_l[1, :] < image.shape[0])
    valid_u_r = (uv_r[0, :] >= 0) & (uv_r[0, :] < image.shape[1])
    valid_v_r = (uv_r[1, :] >= 0) & (uv_r[1, :] < image.shape[0])
    valid_uv = valid_u_l & valid_u_r & valid_v_l & valid_v_r
    ho_mask = np.zeros(uv_l.shape[1], dtype=bool)
    ho_mask[np.asarray(hmax > 0.95, np.int32)] = True
    thresh = 0.5
    if len(std_l.shape) > 1 or len(std_r.shape) > 1:
        std_l = np.std(std_l, axis=1)
        std_r = np.std(std_r, axis=1)

    valid_std = (std_l < thresh) & (std_r < thresh)

    return valid_uv & valid_std & ho_mask


def correl_zscan(points_3d, yaml_file, images_path, Nimg, DEBUG=False, SAVE=True):
    # Paths for yaml file and images

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
    print('Got {} left and right images: \n dt: {} s'.format(right_images.shape[2] + left_images.shape[2],
                                                              round((t1 - t0) , 2)))

    # Read file containing all calibration parameters from stereo system
    Kl, Dl, Rl, Tl, Kr, Dr, Rr, Tr, R, T = rectify_matrix.load_camera_params(yaml_file=yaml_file)

    # Project points on Left and right
    uv_points_l = project_points.gcs2ccs(points_3d, Kl, Dl, Rl, Tl)
    uv_points_r = project_points.gcs2ccs(points_3d, Kr, Dr, Rr, Tr)

    t3 = time.time()
    print("Project points \n dt: {} ms".format(round((t3 - t1) , 2)))

    # Interpolate reprojected points to image bounds (return pixel intensity)
    inter_points_L, std_interp_L = bi_interpolation(left_images, uv_points_l)
    inter_points_R, std_interp_R = bi_interpolation(right_images, uv_points_r)

    t4 = time.time()
    print("Interpolate \n dt: {} ms".format(round((t4 - t3) , 2)))

    # Temporal correlation for L and R interpolated points
    ho, hmax, imax = temp_cross_correlation(inter_points_L, inter_points_R, points_3d)
    # filter_mask = correl_mask(image=left_images, uv_l=uv_points_l, uv_r=uv_points_r,
    #                           std_l=std_interp_L, std_r=std_interp_R, hmax=hmax)

    filtered_3d_ho = points_3d[np.asarray(imax[hmax > 0.9], np.int32)]
    # mask_3d = points_3d[filter_mask]
    t5 = time.time()
    print("Cross Correlation yaml \n dt: {} ms".format(round((t5 - t4) , 2)))



    t6 = time.time()
    print('Ploted 3D points and correlation data \n dt: {} s'.format(round((t6 - t5) , 2)))

    if DEBUG:

        reproj_l = debugger.plot_points_on_image(left_images[:, :, 0], uv_points_l)
        reproj_r = debugger.plot_points_on_image(right_images[:, :, 0], uv_points_r)
        debugger.show_stereo_images(reproj_l, reproj_r, 'Reprojected points 0 image')
        cv2.destroyAllWindows()
        debugger.plot_zscan_correl(ho, xyz_points=points_3d, nimgs=Nimg)

    if SAVE:
        np.savetxt('./correlation_points.txt', filtered_3d_ho, delimiter='\t', fmt='%.3f')

    print('Total time: {} s'.format(round(time.time() - t0, 2)))


    debugger.plot_3d_points(filtered_3d_ho[:, 0], filtered_3d_ho[:, 1], filtered_3d_ho[:, 2], color=None,
                            title="Pearson Correl total for {} images".format(Nimg))
    # debugger.plot_3d_points(mask_3d[:, 0], mask_3d[:, 1], mask_3d[:, 2], color=None,
    #                     title="Mask filter for {} images".format(Nimg))

    print('wait')

def fringe_zscan(points_3d, yaml_file, DEBUG=False, SAVE=True):
    t0 = time.time()

    left_images = []
    right_images = []

    for img_l, img_r in zip(sorted(os.listdir('csv/left')), sorted(os.listdir('csv/right'))):
        left_images.append(debugger.load_array_from_csv(os.path.join('csv/left', img_l)))
        right_images.append(debugger.load_array_from_csv(os.path.join('csv/right', img_r)))

    left_images = np.stack(left_images, axis=-1).astype(np.float32)
    right_images = np.stack(right_images, axis=-1).astype(np.float32)

    t1 = time.time()
    print('Got {} left and right images: \n dt: {} s'.format(right_images.shape[2] + left_images.shape[2],
                                                              round((t1 - t0) , 2)))

    # Read file containing all calibration parameters from stereo system
    Kl, Dl, Rl, Tl, Kr, Dr, Rr, Tr, R, T = rectify_matrix.load_camera_params(yaml_file=yaml_file)

    # Project points on Left and right
    uv_points_l = project_points.gcs2ccs(points_3d, Kl, Dl, Rl, Tl)
    uv_points_r = project_points.gcs2ccs(points_3d, Kr, Dr, Rr, Tr)

    t3 = time.time()
    print('Project points \n dt: {} s'.format(round((t3 - t1), 2)))

    # Interpolate reprojected points to image bounds (return pixel intensity)
    inter_points_L, std_interp_L = bi_interpolation(left_images, uv_points_l)
    inter_points_R, std_interp_R = bi_interpolation(right_images, uv_points_r)

    t4 = time.time()
    print("Interpolate \n dt: {} s".format(round((t4 - t3) , 2)))

    phi_map, phi_min, phi_min_id = phase_map(inter_points_L[:, 0], inter_points_R[:, 0], points_3d)
    # fringe_mask = fringe_masks(image=left_images, uv_l=uv_points_l, uv_r=uv_points_r,
    #                            std_l=std_interp_L, std_r=std_interp_R, phi_id=phi_min_id)

    filtered_3d_phi = points_3d[np.asarray(phi_min_id, np.int32)]
    # filtered_mask = points_3d[fringe_mask]
    if DEBUG:
        debugger.plot_zscan_phi(phi_map=phi_map)

        reproj_l = debugger.plot_points_on_image(left_images[:, :, 0], uv_points_l)
        reproj_r = debugger.plot_points_on_image(right_images[:, :, 0], uv_points_r)
        debugger.show_stereo_images(reproj_l, reproj_r, 'Reprojected points o image')
        cv2.destroyAllWindows()




    if SAVE:
        np.savetxt('./fringe_points.txt', filtered_3d_phi, delimiter='\t', fmt='%.3f')
        # np.savetxt('./fringe_points_fltered.txt', filtered_mask, delimiter='\t', fmt='%.3f')

    print('Total time: {} s'.format(round(time.time() - t0, 2)))
    print('wait')

    debugger.plot_3d_points(filtered_3d_phi[:, 0], filtered_3d_phi[:, 1], filtered_3d_phi[:, 2], color=None,
                            title="Point Cloud of min phase diff")
    # debugger.plot_3d_points(filtered_mask[:, 0], filtered_mask[:, 1], filtered_mask[:, 2], color=None,
    #                     title="Fringe Mask")


def main():
    yaml_file = 'cfg/SM4_20241004_bianca.yaml'
    images_path = 'images/SM4-20241004 - noise'

    points_3d = project_points.points3d_cube(x_lim=(-250, 400), y_lim=(-150, 400), z_lim=(-100, 300),
                                             xy_step=5, z_step=0.1, visualize=False)

    fringe_zscan(points_3d=points_3d, yaml_file=yaml_file, DEBUG=False, SAVE=True)
    correl_zscan(points_3d, yaml_file=yaml_file, images_path=images_path, Nimg=30, DEBUG=False, SAVE=True)


if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
