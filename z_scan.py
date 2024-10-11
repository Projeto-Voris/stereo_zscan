import os
import time
import gc  # Garbage collector
import cv2
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

import debugger
import rectify_matrix
import project_points


def bi_interpolation_gpu(images, uv_points, max_memory_gb=3):
    """
    Perform bilinear interpolation on a stack of images at specified uv_points on the GPU,
    ensuring that each batch uses no more than the specified memory limit.

    Parameters:
    images: (height, width, num_images) array of images, or (height, width) for a single image.
    uv_points: (2, N) array of points where N is the number of points.
    max_memory_gb: Maximum memory to use on the GPU per batch in gigabytes (default 6GB).

    Returns:
    interpolated: (N, num_images) array of interpolated pixel values, or (N,) for a single image.
    std: Standard deviation of the corner pixels used for interpolation.
    """
    images = cp.asarray(images)
    uv_points = cp.asarray(uv_points)

    if len(images.shape) == 2:
        images = images[:, :, cp.newaxis]

    height, width, num_images = images.shape
    N = uv_points.shape[1]

    # Calculate max bytes we can use
    max_bytes = max_memory_gb * 1024 ** 3

    def estimate_memory_for_batch(batch_size):
        bytes_per_float32 = 8
        memory_for_images = height * width * bytes_per_float32  # Only one image at a time
        memory_for_uv = batch_size * 2 * bytes_per_float32
        intermediate_memory = 4 * batch_size * bytes_per_float32  # For p11, p12, etc.
        return memory_for_images + memory_for_uv + intermediate_memory

    batch_size = N
    while estimate_memory_for_batch(batch_size) > max_bytes:
        batch_size //= 2

    # print(f"Batch size adjusted to: {batch_size} to fit within {max_memory_gb}GB GPU memory limit.")

    # Initialize output arrays on CPU to accumulate results
    interpolated_cpu = np.zeros((N, num_images), dtype=np.float32)
    std_cpu = np.zeros((N, num_images), dtype=np.float32)

    # Process points in batches
    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        uv_batch = uv_points[:, i:end]

        x = uv_batch[0].astype(cp.int32)
        y = uv_batch[1].astype(cp.int32)

        x1 = cp.clip(cp.floor(x).astype(cp.int32), 0, width - 1)
        y1 = cp.clip(cp.floor(y).astype(cp.int32), 0, height - 1)
        x2 = cp.clip(x1 + 1, 0, width - 1)
        y2 = cp.clip(y1 + 1, 0, height - 1)

        x_diff = x - x1
        y_diff = y - y1

        for n in range(num_images):  # Process one image at a time
            p11 = images[y1, x1, n]
            p12 = images[y2, x1, n]
            p21 = images[y1, x2, n]
            p22 = images[y2, x2, n]

            interpolated_batch = (
                    p11 * (1 - x_diff) * (1 - y_diff) +
                    p21 * x_diff * (1 - y_diff) +
                    p12 * (1 - x_diff) * y_diff +
                    p22 * x_diff * y_diff
            )

            std_batch = cp.std(cp.vstack([p11, p12, p21, p22]), axis=0)

            interpolated_cpu[i:end, n] = cp.asnumpy(interpolated_batch)
            std_cpu[i:end, n] = cp.asnumpy(std_batch)

        del p11, p12, p21, p22, std_batch, interpolated_batch
        # Free memory after each batch
    # del x1, x2, y1, y2, x2, y2
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()

    if images.shape[2] == 1:
        interpolated_cpu = interpolated_cpu[:, 0]
        std_cpu = std_cpu[:, 0]

    return interpolated_cpu, std_cpu


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
    interpolated = np.empty((N, num_images), dtype=np.float32)
    std = np.empty((N, num_images))

    # Process points in batches
    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        uv_batch = uv_points[:, i:end]

        x = uv_batch[0].astype(np.int32)
        y = uv_batch[1].astype(np.int32)

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
            std_batch = np.std(np.vstack([p11, p12, p21, p22]), axis=0)
            std[i:end, n] = std_batch

    # Return 1D interpolated result if the input was a 2D image
    if images.shape[2] == 1:
        interpolated = interpolated[:, 0]
        std = std[:, 0]

    return interpolated, std


def temp_cross_correlation(left_Igray, right_Igray, points_3d):
    """
    Calculate the cross-correlation between two sets of images over time in chunks,
    focusing on minimizing memory usage.

    Parameters:
    left_Igray: (num_points, num_images) array of left images in grayscale.
    right_Igray: (num_points, num_images) array of right images in grayscale.
    points_3d: (num_points, 3) array of 3D points.

    Returns:
    ho: Cross-correlation values.
    hmax: Maximum correlation values.
    Imax: Indices of maximum correlation values.
    ho_ztep: List of correlation values for all Z value of each XY.
    """
    # Convert images to float32 if they aren't already
    left_Igray = left_Igray.astype(np.float32)
    right_Igray = right_Igray.astype(np.float32)

    num_images = left_Igray.shape[1]  # Number of images
    num_points = left_Igray.shape[0]  # Number of points

    # Number of tested Z (this will be used as batch size)
    z_size = np.unique(points_3d[:, 2]).shape[0]
    num_pairs = points_3d.shape[0] // z_size  # Total number of XY points

    # Initialize outputs with the correct data type (float32 for memory efficiency)
    ho = np.empty(num_points, dtype=np.float32)
    hmax = np.empty(num_pairs, dtype=np.float32)
    Imax = np.empty(num_pairs, dtype=np.int64)

    # Preallocate ho_ztep only if necessary (otherwise remove this)
    ho_ztep = np.empty((z_size, num_pairs), dtype=np.float32)  # Store values for each Z value tested per XY pair

    # Process images in chunks based on z_size (number of Z values)
    for k in range(num_pairs):
        start_idx = k * z_size
        end_idx = (k + 1) * z_size

        # Mean values along time (current XY point across all Z values)
        left_mean_batch = np.mean(left_Igray[start_idx:end_idx, :], axis=1, keepdims=True)
        right_mean_batch = np.mean(right_Igray[start_idx:end_idx, :], axis=1, keepdims=True)

        # Calculate the numerator and denominator for the correlation
        num = np.sum((left_Igray[start_idx:end_idx, :] - left_mean_batch) *
                     (right_Igray[start_idx:end_idx, :] - right_mean_batch), axis=1)
        left_sq_diff = np.sum((left_Igray[start_idx:end_idx, :] - left_mean_batch) ** 2, axis=1)
        right_sq_diff = np.sum((right_Igray[start_idx:end_idx, :] - right_mean_batch) ** 2, axis=1)

        den = np.sqrt(left_sq_diff * right_sq_diff)
        ho_batch = num / np.maximum(den, 1e-10)

        # Ensure ho_batch matches the size of the destination
        if ho_batch.shape[0] != (end_idx - start_idx):
            raise ValueError(f"Shape mismatch: ho_batch has shape {ho_batch.shape}, expected {(end_idx - start_idx)}")

        # Store the results for this batch (ensure correct sizing)
        ho[start_idx:end_idx] = ho_batch  # Only assign correct range

        # Extract and calculate ho_ztep, hmax, Imax for this XY point
        # ho_ztep[:,k] = ho_range  # Store Z-step correlation values (if needed)
        hmax[k] = np.nanmax(ho_batch)
        Imax[k] = np.nanargmax(ho_batch) + start_idx  # Add offset to get the index in the full array

        # Release memory after processing each batch
        del left_mean_batch, right_mean_batch, ho_batch
        gc.collect()

    return ho, hmax, Imax, ho_ztep

def temp_cross_correlation_gpu(left_Igray, right_Igray, points_3d, max_memory_gb=3):
    """
    Calculate the cross-correlation between two sets of images over time using CuPy for GPU acceleration,
    while limiting GPU memory usage and handling variable batch sizes.

    Parameters:
    left_Igray: (num_points, num_images) array of left images in grayscale.
    right_Igray: (num_points, num_images) array of right images in grayscale.
    points_3d: (num_points, 3) array of 3D points.
    max_memory_gb: Maximum GPU memory to use in GB (default is 4GB).

    Returns:
    ho: Cross-correlation values.
    hmax: Maximum correlation values.
    Imax: Indices of maximum correlation values.
    ho_ztep: List of correlation values for all Z value of each XY.
    """

    # Convert images to CuPy arrays for GPU computation
    left_Igray = cp.asarray(left_Igray, dtype=cp.float32)
    right_Igray = cp.asarray(right_Igray, dtype=cp.float32)

    num_images = left_Igray.shape[1]  # Number of images
    num_points = left_Igray.shape[0]  # Number of points

    # Number of tested Z (this will be used as batch size)
    z_size = cp.unique(points_3d[:, 2]).shape[0]
    num_pairs = points_3d.shape[0] // z_size  # Total number of XY points

    # Estimate memory usage per point
    bytes_per_float32 = 8
    memory_per_point = (2 * num_images * bytes_per_float32)  # For left_Igray, right_Igray, and intermediate calculations
    total_memory_required = num_points * memory_per_point

    # Maximum bytes allowed for memory usage
    max_bytes = max_memory_gb * 1024**3

    # Adjust the batch size based on memory limitations
    if total_memory_required > max_bytes:
        points_per_batch = int(max_bytes // memory_per_point // 10)
        # print(f"Processing {points_per_batch} points per batch due to memory limitations.")
    else:
        points_per_batch = num_points  # Process all points at once

    # Initialize outputs with the correct data type (float32 for memory efficiency)
    ho = cp.empty(num_points, dtype=cp.float32)
    hmax = cp.empty(num_pairs, dtype=cp.float32)
    Imax = cp.empty(num_pairs, dtype=cp.int64)

    # Preallocate ho_ztep only if necessary
    ho_ztep = cp.empty((z_size, num_pairs), dtype=cp.float32)  # Store values for each Z value tested per XY pair

    # Process images in chunks based on the adjusted points_per_batch size
    for i in range(0, num_points, points_per_batch):
        end = min(i + points_per_batch, num_points)
        batch_left = left_Igray[i:end]
        batch_right = right_Igray[i:end]
        batch_size = batch_left.shape[0]

        # Debug: Check the batch size
        # print(f"Processing batch {i // points_per_batch + 1}, size: {batch_size}")

        # Mean values along time (for the current batch)
        left_mean_batch = cp.mean(batch_left, axis=1, keepdims=True)
        right_mean_batch = cp.mean(batch_right, axis=1, keepdims=True)

        # Calculate the numerator and denominator for the correlation
        num = cp.sum((batch_left - left_mean_batch) * (batch_right - right_mean_batch), axis=1)
        left_sq_diff = cp.sum((batch_left - left_mean_batch) ** 2, axis=1)
        right_sq_diff = cp.sum((batch_right - right_mean_batch) ** 2, axis=1)

        den = cp.sqrt(left_sq_diff * right_sq_diff)
        ho_batch = num / cp.maximum(den, 1e-10)

        # Store the results for this batch
        ho[i:end] = ho_batch

        # Calculate hmax and Imax for this batch
        for k in range(batch_size // z_size):
            batch_start_idx = k * z_size
            batch_end_idx = (k + 1) * z_size
            ho_range = ho_batch[batch_start_idx:batch_end_idx]

            hmax[i // z_size + k] = cp.nanmax(ho_range)
            Imax[i // z_size + k] = cp.nanargmax(ho_range) + i

        # Release memory after processing each batch
        del batch_left, batch_right, left_mean_batch, right_mean_batch, ho_batch
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()

    # Handle the case where the last batch is smaller than others
    # print(f"Handling final batch (size: {batch_size}) to avoid dimension mismatch.")
    # No need for padding as we're concatenating batch-wise

    return cp.asnumpy(ho), cp.asnumpy(hmax), cp.asnumpy(Imax), cp.asnumpy(ho_ztep)


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


def fringe_masks(image, uv_l, uv_r, std_l, std_r, phi_id, min_thresh, max_thresh):
    valid_u_l = (uv_l[0, :] >= 0) & (uv_l[0, :] < image.shape[1])
    valid_v_l = (uv_l[1, :] >= 0) & (uv_l[1, :] < image.shape[0])
    valid_u_r = (uv_r[0, :] >= 0) & (uv_r[0, :] < image.shape[1])
    valid_v_r = (uv_r[1, :] >= 0) & (uv_r[1, :] < image.shape[0])
    valid_uv = valid_u_l & valid_u_r & valid_v_l & valid_v_r
    phi_mask = np.zeros(uv_l.shape[1], dtype=bool)
    phi_mask[phi_id] = True
    valid_std = (min_thresh < std_l < max_thresh) & (min_thresh < std_r < max_thresh)

    return valid_uv & valid_std & phi_mask


def correl_mask(image, uv_l, uv_r, std_l, std_r, hmax, min_thresh, max_thresh):
    valid_u_l = (uv_l[0, :] >= 0) & (uv_l[0, :] < image.shape[1])
    valid_v_l = (uv_l[1, :] >= 0) & (uv_l[1, :] < image.shape[0])
    valid_u_r = (uv_r[0, :] >= 0) & (uv_r[0, :] < image.shape[1])
    valid_v_r = (uv_r[1, :] >= 0) & (uv_r[1, :] < image.shape[0])
    valid_uv = valid_u_l & valid_u_r & valid_v_l & valid_v_r
    ho_mask = np.zeros(uv_l.shape[1], dtype=bool)
    ho_mask[np.asarray(hmax > 0.95, np.int32)] = True
    if len(std_l.shape) > 1 or len(std_r.shape) > 1:
        std_l = np.std(std_l, axis=1)
        std_r = np.std(std_r, axis=1)

    valid_std = (min_thresh < std_l < max_thresh) & (min_thresh < std_r < max_thresh)

    return valid_uv & valid_std & ho_mask


def correl_zscan(points_3d, yaml_file, images_path, Nimg, output='Correl_pts', DEBUG=False, SAVE=True):
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
                                                             round((t1 - t0), 2)))

    # Read file containing all calibration parameters from stereo system
    Kl, Dl, Rl, Tl, Kr, Dr, Rr, Tr, R, T = rectify_matrix.load_camera_params(yaml_file=yaml_file)

    # Project points on Left and right
    uv_points_l = project_points.gcs2ccs_gpu(points_3d, Kl, Dl, Rl, Tl)
    uv_points_r = project_points.gcs2ccs_gpu(points_3d, Kr, Dr, Rr, Tr)

    t3 = time.time()
    print("Project points \n dt: {} s".format(round((t3 - t1), 2)))

    # Interpolate reprojected points to image bounds GPU
    inter_points_L, std_interp_L = bi_interpolation_gpu(left_images, uv_points_l)
    inter_points_R, std_interp_R = bi_interpolation_gpu(right_images, uv_points_r)

    t4 = time.time()
    print("Interpolate GPU \n dt: {} s".format(round((t4 - t3), 2)))

    # Temporal correlation for L and R interpolated points
    ho, hmax, imax, ho_zstep = temp_cross_correlation_gpu(inter_points_L, inter_points_R, points_3d)
    # filter_mask = correl_mask(image=left_images, uv_l=uv_points_l, uv_r=uv_points_r,
    #                           std_l=std_interp_L, std_r=std_interp_R, hmax=hmax)

    # mask_3d = points_3d[filter_mask]
    t5 = time.time()
    print("Cross Correlation \n dt: {} s".format(round((t5 - t4), 2)))

    # Temporal correlation for L and R interpolated points
    # ho, hmax, imax, ho_zstep = temp_cross_correlation(inter_points_L, inter_points_R, points_3d)
    # filter_mask = correl_mask(image=left_images, uv_l=uv_points_l, uv_r=uv_points_r,
    #                           std_l=std_interp_L, std_r=std_interp_R, hmax=hmax)

    # mask_3d = points_3d[filter_mask]
    print("Cross Correlation \n dt: {} s".format(round((time.time() - t5), 2)))
    filtered_3d_ho = points_3d[np.asarray(imax[hmax > 0.95], np.int32)]

    t6 = time.time()
    print('Ploted 3D points and correlation data \n dt: {} s'.format(round((t6 - t5), 2)))

    if DEBUG:
        reproj_l = debugger.plot_points_on_image(left_images[:, :, 0], uv_points_l)
        reproj_r = debugger.plot_points_on_image(right_images[:, :, 0], uv_points_r)
        debugger.show_stereo_images(reproj_l, reproj_r, 'Reprojected points 0 image')
        cv2.destroyAllWindows()
        debugger.plot_zscan_correl(ho, xyz_points=points_3d, nimgs=Nimg)

    if SAVE:
        np.savetxt('./points_correl_{}_imgs_{}.txt'.format(Nimg, output), filtered_3d_ho, delimiter='\t', fmt='%.3f')

    print('Total time: {} s'.format(round(time.time() - t0, 2)))

    debugger.plot_3d_points(filtered_3d_ho[:, 0], filtered_3d_ho[:, 1], filtered_3d_ho[:, 2],
                            color=hmax[hmax > 0.95],
                            title="Pearson Correl total for {} images from {}".format(Nimg, output))
    # debugger.plot_3d_points(mask_3d[:, 0], mask_3d[:, 1], mask_3d[:, 2], color=None,
    #                     title="Mask filter for {} images".format(Nimg))

    print('wait')


def fringe_zscan(points_3d, yaml_file, image_name, output='fringe_points', DEBUG=False, SAVE=True):
    t0 = time.time()

    left_images = []
    right_images = []

    for img_l, img_r in zip(sorted(os.listdir('csv/left')), sorted(os.listdir('csv/right'))):
        if img_l.split('left_abs_')[-1] == image_name:
            left_images.append(debugger.load_array_from_csv(os.path.join('csv/left', img_l)))
        if img_r.split('right_abs_')[-1] == image_name:
            right_images.append(debugger.load_array_from_csv(os.path.join('csv/right', img_r)))

    left_images = np.stack(left_images, axis=-1).astype(np.float16)
    right_images = np.stack(right_images, axis=-1).astype(np.float16)

    t1 = time.time()
    print('Got {} left and right images: \n dt: {} s'.format(right_images.shape[2] + left_images.shape[2],
                                                             round((t1 - t0), 2)))

    # Read file containing all calibration parameters from stereo system
    Kl, Dl, Rl, Tl, Kr, Dr, Rr, Tr, R, T = rectify_matrix.load_camera_params(yaml_file=yaml_file)

    # Project points on Left and right
    uv_points_l = project_points.gcs2ccs_gpu(points_3d, Kl, Dl, Rl, Tl)
    uv_points_r = project_points.gcs2ccs_gpu(points_3d, Kr, Dr, Rr, Tr)

    t3 = time.time()
    print('Project points \n dt: {} s'.format(round((t3 - t1), 2)))

    # Interpolate reprojected points to image bounds (return pixel intensity)
    inter_points_L, std_interp_L = bi_interpolation_gpu(left_images, uv_points_l)
    inter_points_R, std_interp_R = bi_interpolation_gpu(right_images, uv_points_r)

    t4 = time.time()
    print("Interpolate \n dt: {} s".format(round((t4 - t3), 2)))

    phi_map, phi_min, phi_min_id = phase_map(inter_points_L, inter_points_R, points_3d)
    # fringe_mask = fringe_masks(image=left_images, uv_l=uv_points_l, uv_r=uv_points_r,
    #                            std_l=std_interp_L, std_r=std_interp_R, phi_id=phi_min_id)
    t5 = time.time()
    print("Phase map \n dt: {} s".format(round((t5 - t3), 2)))
    filtered_3d_phi = points_3d[np.asarray(phi_min_id, np.int32)]
    # filtered_mask = points_3d[fringe_mask]
    if DEBUG:
        debugger.plot_zscan_phi(phi_map=phi_map)

        reproj_l = debugger.plot_points_on_image(left_images[:, :, 0], uv_points_l)
        reproj_r = debugger.plot_points_on_image(right_images[:, :, 0], uv_points_r)
        debugger.show_stereo_images(reproj_l, reproj_r, 'Reprojected points o image')
        cv2.destroyAllWindows()

    if SAVE:
        np.savetxt('./fringe_points_{}_pxf.txt'.format(output), filtered_3d_phi, delimiter='\t', fmt='%.3f')
        # np.savetxt('./fringe_points_fltered.txt', filtered_mask, delimiter='\t', fmt='%.3f')

    print('Total time: {} s'.format(round(time.time() - t0, 2)))
    print('wait')

    debugger.plot_3d_points(filtered_3d_phi[:, 0], filtered_3d_phi[:, 1], filtered_3d_phi[:, 2], color=None,
                            title="Point Cloud of min phase diff - {} px per fringe".format(output))
    # debugger.plot_3d_points(filtered_mask[:, 0], filtered_mask[:, 1], filtered_mask[:, 2], color=None,
    #                     title="Fringe Mask")


def main():
    yaml_file_SM4 = 'cfg/SM4_20241004_bianca.yaml'
    images_path_SM4 = 'images/SM4-20241004 - noise'  # sm4
    yaml_file_SM3 = 'cfg/SM3_20240918_bouget.yaml'
    # images_path_SM3 = 'images/SM3-20240918 - noise'
    images_path_SM3 = '/home/daniel/Insync/daniel.regner@labmetro.ufsc.br/Google Drive - Shared drives/VORIS  - Equipe/Sistema de Medição 3 - Stereo Ativo - Projeção Laser/Imagens/Testes/SM3-20240828 - GC (f50)'
    fringe_image_name = '016.csv'

    t0 = time.time()
    points_3d_SM4 = project_points.points3d_cube(x_lim=(-50, 250), y_lim=(-150, 200), z_lim=(-100, 300),
                                                 xy_step=2, z_step=0.1, visualize=False)  # pontos para SM4
    t1 = time.time()
    print('Time for 3d points {} dt'.format(round((t1 - t0), 2)))
    # points_3d_SM4 = project_points.points3d_cube_gpu(x_lim=(-50, 250), y_lim=(-150, 200), z_lim=(-100, 300),
    #                                              xy_step=2, z_step=0.1, visualize=False)  # pontos para SM4
    #
    # print('Time for 3d points gpu {} dt'.format(round((time.time() - t1), 2)))

    points_3d_SM3 = project_points.points3d_cube(x_lim=(-250, 300), y_lim=(-250, 250), z_lim=(-100, 100),
                                                 xy_step=5, z_step=0.1, visualize=False)  # pontos para SM3

    # fringe_zscan(points_3d=points_3d_SM4, yaml_file=yaml_file_SM4, image_name=fringe_image_name,
    #              output=fringe_image_name.split('.')[0], DEBUG=False, SAVE=False)
    #
    correl_zscan(points_3d_SM4, yaml_file=yaml_file_SM4, images_path=images_path_SM4, Nimg=20,
                 output=images_path_SM3.split('/')[-1], DEBUG=False, SAVE=False)


if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
