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
    num_points = uv_points.shape[1]

    # Estimate memory usage per point
    bytes_per_float32 = 8
    memory_per_point = (4 * num_images * bytes_per_float32)
    # For left_Igray, right_Igray, and intermediate calculations
    total_memory_required = num_points * memory_per_point

    # Maximum bytes allowed for memory usage
    max_bytes = max_memory_gb * 1024 ** 3

    # Adjust the batch size based on memory limitations
    if total_memory_required > max_bytes:
        points_per_batch = int(max_bytes // memory_per_point // 100)
        # print(f"Processing {points_per_batch} points per batch due to memory limitations.")
    else:
        points_per_batch = num_points  # Process all points at once

    # print(f"Batch size adjusted to: {batch_size} to fit within {max_memory_gb}GB GPU memory limit.")

    # Initialize output arrays on CPU to accumulate results
    interpolated_cpu = np.empty((num_points, num_images), dtype=np.float16)
    std_cpu = np.empty((num_points, num_images), dtype=np.float16)

    # Process points in batches
    for i in range(0, num_points, points_per_batch):
        end = min(i + points_per_batch, num_points)
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

    num_images = left_Igray.shape[1]  # Number of images
    num_points = left_Igray.shape[0]  # Number of points

    # Number of tested Z (this will be used as batch size)
    z_size = cp.unique(points_3d[:, 2]).shape[0]
    num_pairs = points_3d.shape[0] // z_size  # Total number of XY points

    # Estimate memory usage per point
    bytes_per_float32 = 8
    memory_per_point = (4 * num_images * bytes_per_float32)
    # For left_Igray, right_Igray, and intermediate calculations
    total_memory_required = num_points * memory_per_point

    # Maximum bytes allowed for memory usage
    max_bytes = max_memory_gb * 1024 ** 3

    # Adjust the batch size based on memory limitations
    if total_memory_required > max_bytes:
        points_per_batch = int(max_bytes // memory_per_point // 10)
        # print(f"Processing {points_per_batch} points per batch due to memory limitations.")
    else:
        points_per_batch = num_points  # Process all points at once

    # Initialize outputs with the correct data type (float32 for memory efficiency)
    ho = cp.empty(num_points, dtype=cp.float32)

    # Preallocate ho_ztep only if necessary
    ho_ztep = cp.empty((z_size, num_pairs), dtype=cp.float32)  # Store values for each Z value tested per XY pair

    # Process images in chunks based on the adjusted points_per_batch size
    for i in range(0, num_points, points_per_batch):
        end = min(i + points_per_batch, num_points)

        # Load only the current batch into the GPU
        batch_left = cp.asarray(left_Igray[i:end], dtype=cp.float32)
        batch_right = cp.asarray(right_Igray[i:end], dtype=cp.float32)

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

        # Release memory after processing each batch
        del batch_left, batch_right, left_mean_batch, right_mean_batch, ho_batch
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()

    hmax = cp.empty(num_pairs, dtype=cp.float32)
    Imax = cp.empty(num_pairs, dtype=cp.int64)
    # Calculate hmax and Imax for this batch
    for k in range(points_3d.shape[0] // z_size):
        start_idx = k * z_size
        end_idx = (k + 1) * z_size
        ho_range = ho[start_idx:end_idx]

        hmax[k] = cp.nanmax(ho_range)
        Imax[k] = cp.nanargmax(ho_range) + k * z_size

    return ho, hmax, Imax, ho_ztep


def spatial_correlation(images_left, images_right, uv_points_l, uv_points_r, points_3d, window_size=3):
    """
    Vectorized computation of spatial correlation for patches around specified points across all images at once.

    Parameters:
    images_left: (height, width, num_images) array of the left image stack.
    images_right: (height, width, num_images) array of the right image stack.
    uv_points_l: (2, N) array of UV points for the left image.
    uv_points_r: (2, N) array of UV points for the right image.
    points_3d: (N,3) array of XYZ combination points of GCS
    window_size: Size of the patch window around each point (default is 3).

    Returns:
    spatial_corr: (N,) array of spatial correlation values for each point across all images.
    """

    half_window = window_size // 2
    height, width, num_images = images_left.shape
    num_points = uv_points_l.shape[1]

    # Allocate space for the result
    spatial_corr = np.zeros(num_points, dtype=np.float32)

    # Vectorized window bounds for left and right points
    x1_l = np.clip(uv_points_l[0] - half_window, 0, width - window_size).astype(np.int32)
    y1_l = np.clip(uv_points_l[1] - half_window, 0, height - window_size).astype(np.int32)

    x1_r = np.clip(uv_points_r[0] - half_window, 0, width - window_size).astype(np.int32)
    y1_r = np.clip(uv_points_r[1] - half_window, 0, height - window_size).astype(np.int32)

    # Constructing indices for batch-based patch extraction
    # Left patches: Shape (batch_size, window_size, window_size, num_images)
    patch_left = np.array([images_left[y:y + window_size, x:x + window_size, :]
                           for y, x in zip(y1_l, x1_l)], dtype=np.float32)

    # Right patches: Shape (batch_size, window_size, window_size, num_images)
    patch_right = np.array([images_right[y:y + window_size, x:x + window_size, :]
                            for y, x in zip(y1_r, x1_r)], dtype=np.float32)

    # Vectorized mean subtraction
    mean_left = np.mean(patch_left, axis=(1, 2), keepdims=True)
    mean_right = np.mean(patch_right, axis=(1, 2), keepdims=True)

    patch_left -= mean_left
    patch_right -= mean_right

    # Compute numerator and denominator using vectorized operations
    num = np.sum(patch_left * patch_right, axis=(1, 2))
    den_left = np.sum(patch_left ** 2, axis=(1, 2))
    den_right = np.sum(patch_right ** 2, axis=(1, 2))

    # Compute the spatial correlation for all points in the batch
    num_total = np.sum(num, axis=1)
    den_total = np.sqrt(np.sum(den_left, axis=1) * np.sum(den_right, axis=1))

    # Final spatial correlation for all points
    spatial_corr = num_total / (den_total + 1e-10)  # Avoid division by zero
    z_step = np.unique(points_3d[:, 2]).shape[0]
    spatial_max = np.empty(points_3d.shape[0] // z_step, dtype=np.float32)
    spatial_id = np.empty(points_3d.shape[0] // z_step, dtype=np.int32)
    for k in range(points_3d.shape[0] // z_step):
        spatial_range = spatial_corr[k * z_step:(k + 1) * z_step]
        spatial_max[k] = np.nanmax(spatial_range)
        spatial_id[k] = np.nanargmax(spatial_range) + k * z_step

    return spatial_corr, spatial_max, spatial_id
def spatial_correlation_gpu(images_left, images_right, uv_points_l, uv_points_r, points_3d, window_size=3,
                            max_memory_gb=3):
    """
      Optimized GPU-accelerated computation of spatial correlation with reduced memory overhead and optimized batch processing.

      Parameters:
      images_left: (height, width, num_images) array of the left image stack (on GPU).
      images_right: (height, width, num_images) array of the right image stack (on GPU).
      uv_points_l: (2, N) array of UV points for the left image (on GPU).
      uv_points_r: (2, N) array of UV points for the right image (on GPU).
      points_3d: (N, 3) array of XYZ combination points of GCS (on GPU).
      window_size: Size of the patch window around each point (default is 3).
      max_memory_gb: Maximum GPU memory usage in GB.

      Returns:
      spatial_corr: (N,) array of spatial correlation values for each point across all images (on GPU).
      spatial_max: (N/z_step,) array of max spatial correlation values for each point group (on GPU).
      spatial_id: (N/z_step,) array of indices of max spatial correlation values for each point group (on GPU).
      """
    half_window = window_size // 2
    height, width, num_images = images_left.shape
    num_points = uv_points_l.shape[1]

    # Estimate memory usage per point
    bytes_per_float32 = 4
    memory_per_point = (2 * window_size * window_size * num_images * bytes_per_float32)  # For left and right patches
    total_memory_required = num_points * memory_per_point

    # Maximum allowed memory in bytes
    max_bytes = max_memory_gb * 1024 ** 3

    # Adjust the batch size based on memory limitations
    points_per_batch = max(1, int(max_bytes // memory_per_point))

    # Allocate space for the result on GPU
    spatial_corr = cp.zeros(num_points, dtype=cp.float32)

    for i in range(0, num_points, points_per_batch):
        end_idx = min(i + points_per_batch, num_points)
        uv_batch_l = uv_points_l[:, i:end_idx]
        uv_batch_r = uv_points_r[:, i:end_idx]

        # Vectorized window bounds for left and right points
        x1_l = cp.clip(uv_batch_l[0] - half_window, 0, width - window_size).astype(cp.int32)
        y1_l = cp.clip(uv_batch_l[1] - half_window, 0, height - window_size).astype(cp.int32)

        x1_r = cp.clip(uv_batch_r[0] - half_window, 0, width - window_size).astype(cp.int32)
        y1_r = cp.clip(uv_batch_r[1] - half_window, 0, height - window_size).astype(cp.int32)

        # Extract patches for left and right images across all images in the batch
        patch_left = cp.array([images_left[y:y + window_size, x:x + window_size, :]
                               for y, x in zip(y1_l, x1_l)], dtype=cp.float32)

        patch_right = cp.array([images_right[y:y + window_size, x:x + window_size, :]
                                for y, x in zip(y1_r, x1_r)], dtype=cp.float32)

        # Vectorized mean subtraction
        mean_left = cp.mean(patch_left, axis=(1, 2), keepdims=True)
        mean_right = cp.mean(patch_right, axis=(1, 2), keepdims=True)

        patch_left -= mean_left
        patch_right -= mean_right

        # Compute numerator and denominator using vectorized operations
        num = cp.sum(patch_left * patch_right, axis=(1, 2))
        den_left = cp.sum(patch_left ** 2, axis=(1, 2))
        den_right = cp.sum(patch_right ** 2, axis=(1, 2))

        # Compute the spatial correlation for all points in the batch
        num_total = cp.sum(num, axis=1)
        den_total = cp.sqrt(cp.sum(den_left, axis=1) * cp.sum(den_right, axis=1))

        # Final spatial correlation for the batch
        spatial_corr[i:end_idx] = num_total / (den_total + 1e-10)  # Avoid division by zero

    # Process spatial max and spatial ID vectorized
    z_step = cp.unique(points_3d[:, 2]).shape[0]
    reshaped_corr = spatial_corr.reshape((-1, z_step))
    spatial_max = cp.nanmax(reshaped_corr, axis=1)
    spatial_id = cp.nanargmax(reshaped_corr, axis=1) + cp.arange(reshaped_corr.shape[0]) * z_step

    # Return results as NumPy arrays
    return cp.asnumpy(spatial_corr), cp.asnumpy(spatial_max), cp.asnumpy(spatial_id)

def fuse_correlations(spatial_corr, temporal_corr, points_3d, alpha=0.5, beta=0.5):
    """
    Fuse spatial and temporal correlations using a weighted sum.

    Parameters:
    spatial_corr: (N, num_images) array of spatial correlation values for each point across images.
    temporal_corr: (N,) array of temporal correlation values for each point.
    alpha: Weight for spatial correlation.
    beta: Weight for temporal correlation.

    Returns:
    fused_corr: (N,) array of fused correlation values for each point.
    """
    # Normalize spatial correlations (optional)
    spatial_corr_norm = spatial_corr / (np.max(spatial_corr) + 1e-10)

    # Normalize temporal correlation (optional)
    temporal_corr_norm = temporal_corr / (np.max(temporal_corr) + 1e-10)

    # Combine the correlations using a weighted sum
    fused_corr = alpha * spatial_corr_norm + beta * temporal_corr_norm  # Reshape for broadcasting

    z_step = np.unique(points_3d[:, 2]).shape[0]

    correl_max = np.empty((points_3d.shape[0] // z_step), dtype=np.float32)
    correl_id = np.empty((points_3d.shape[0] // z_step), dtype=np.float32)
    for k in range(points_3d.shape[0] // z_step):
        correl_range = fused_corr[k * z_step:(k + 1) * z_step]
        correl_max[k] = np.nanmax(correl_range)
        correl_id[k] = np.nanargmax(correl_range) + k * z_step

    return fused_corr, correl_max, correl_id


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


def fringe_masks(image, uv_l, uv_r, std_l, std_r, phi_id, min_thresh=0, max_thresh=1):
    valid_u_l = (uv_l[0, :] >= 0) & (uv_l[0, :] < image.shape[1])
    valid_v_l = (uv_l[1, :] >= 0) & (uv_l[1, :] < image.shape[0])
    valid_u_r = (uv_r[0, :] >= 0) & (uv_r[0, :] < image.shape[1])
    valid_v_r = (uv_r[1, :] >= 0) & (uv_r[1, :] < image.shape[0])
    valid_uv = valid_u_l & valid_u_r & valid_v_l & valid_v_r
    phi_mask = np.zeros(uv_l.shape[1], dtype=bool)
    phi_mask[phi_id] = True
    valid_l = (min_thresh < std_l) & (std_l < max_thresh)
    valid_r = (min_thresh < std_r) & (std_r < max_thresh)

    valid_std = valid_r & valid_l

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


def correl_zscan(points_3d, yaml_file, images_path, Nimg, win_size=7, output='Correl_pts', DEBUG=False, SAVE=True):
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
    print("Interpolate \n dt: {} s".format(round((t4 - t3), 2)))
    if win_size <= 5:
        spatial_corr, spatial_max, spatial_id = spatial_correlation(images_left=left_images, images_right=right_images,
                                                                    uv_points_l=uv_points_l, uv_points_r=uv_points_r,
                                                                    points_3d=points_3d, window_size=win_size)
    else:
        spatial_corr, spatial_max, spatial_id = spatial_correlation_gpu(images_left=left_images,
                                                                                    images_right=right_images,
                                                                                    uv_points_l=uv_points_l,
                                                                                    uv_points_r=uv_points_r,
                                                                                    points_3d=points_3d,
                                                                                    window_size=win_size)

    t5 = time.time()
    print("Spatial GPU\n dt: {} s".format(round((t5 - t4), 2)))



    # Temporal correlation for L and R interpolated points
    ho, hmax, imax, ho_zstep = temp_cross_correlation_gpu(inter_points_L, inter_points_R, points_3d)

    t6 = time.time()
    print("Cross Correlation \n dt: {} s".format(round((t6 - t5), 2)))

    fused_corr, fused_max, fused_id = fuse_correlations(spatial_corr=spatial_corr, temporal_corr=cp.asnumpy(ho),
                                                        points_3d=points_3d, alpha=0.7, beta=0.3)

    t7 = time.time()
    print("Fused Correlation\n dt: {} s".format(round((t7 - t6), 2)))

    filtered_3d_ho = points_3d[cp.asnumpy(imax[hmax > 0.95]).astype(np.int32)]
    filtered_3d_fused = points_3d[np.asarray(fused_id[fused_max > 0.95]).astype(np.int32)]
    filtered_3d_spatial = points_3d[np.asarray(spatial_id[spatial_max > 0.95]).astype(np.int32)]

    print('Total time: {} s'.format(round(time.time() - t0, 2)))

    if DEBUG:
        reproj_l = debugger.plot_points_on_image(left_images[:, :, 0], uv_points_l)
        reproj_r = debugger.plot_points_on_image(right_images[:, :, 0], uv_points_r)
        debugger.show_stereo_images(reproj_l, reproj_r, 'Reprojected points 0 image')
        cv2.destroyAllWindows()
        debugger.plot_zscan_correl(ho, xyz_points=points_3d, nimgs=Nimg)

    if SAVE:
        np.savetxt('./points_correl_{}_imgs_{}.txt'.format(Nimg, output), filtered_3d_ho, delimiter='\t', fmt='%.3f')

    debugger.plot_3d_points(filtered_3d_ho[:, 0], filtered_3d_ho[:, 1], filtered_3d_ho[:, 2],
                            color=cp.asnumpy(hmax[hmax > 0.95].astype(np.float32)),
                            title="Temporal Correl total for {} images and {} window size,from {}".format(Nimg,win_size, output))

    debugger.plot_3d_points(filtered_3d_spatial[:, 0], filtered_3d_spatial[:, 1], filtered_3d_spatial[:, 2],
                            color=spatial_max[spatial_max > 0.95],
                            title="Spatial Correl total for {} images and {} window size,from {}".format(Nimg,win_size, output))

    debugger.plot_3d_points(filtered_3d_fused[:, 0], filtered_3d_fused[:, 1], filtered_3d_fused[:, 2],
                            color=fused_max[fused_max > 0.95].astype(np.float32),
                            title="Fused Correl total for {} images and {} window size,from {}".format(Nimg,win_size, output))

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
    fringe_mask = fringe_masks(image=left_images, uv_l=uv_points_l, uv_r=uv_points_r,
                               std_l=std_interp_L, std_r=std_interp_R, phi_id=phi_min_id)
    t5 = time.time()
    print("Phase map \n dt: {} s".format(round((t5 - t3), 2)))
    filtered_3d_phi = points_3d[np.asarray(phi_min_id, np.int32)]
    filtered_mask = points_3d[fringe_mask]
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
    debugger.plot_3d_points(filtered_mask[:, 0], filtered_mask[:, 1], filtered_mask[:, 2], color=None,
                            title="Fringe Mask")


def main():
    yaml_file_SM4 = 'cfg/SM4_20241004_bianca.yaml'
    images_path_SM4 = 'images/SM4-20241004 - noise'  # sm4
    yaml_file_SM3 = 'cfg/SM3_20240918_bouget.yaml'
    images_path_SM3 = 'images/SM3-20240918 - noise'
    # images_path_SM3 = '/home/daniel/Insync/daniel.regner@labmetro.ufsc.br/Google Drive - Shared drives/VORIS  - Equipe/Sistema de Medição 3 - Stereo Ativo - Projeção Laser/Imagens/Testes/SM3-20240828 - GC (f50)'
    fringe_image_name = '016.csv'

    t0 = time.time()
    points_3d_SM4 = project_points.points3d_cube(x_lim=(-50, 250), y_lim=(-150, 200), z_lim=(-100, 200),
                                                 xy_step=5, z_step=0.1, visualize=False)  # pontos para SM4
    print('Time for 3d points \n {} dt'.format(round((time.time() - t0), 2)))
    # points_3d_SM4 = project_points.points3d_cube_gpu(x_lim=(-50, 250), y_lim=(-150, 200), z_lim=(-100, 300),
    #                                              xy_step=2, z_step=0.1, visualize=False)  # pontos para SM4
    #
    # print('Time for 3d points gpu {} dt'.format(round((time.time() - t1), 2)))

    points_3d_SM3 = project_points.points3d_cube(x_lim=(-250, 300), y_lim=(-250, 250), z_lim=(-100, 100),
                                                 xy_step=5, z_step=0.1, visualize=False)  # pontos para SM3

    # fringe_zscan(points_3d=points_3d_SM4, yaml_file=yaml_file_SM4, image_name=fringe_image_name,
    #              output=os.path.join('output', fringe_image_name.split('.')[0]), DEBUG=False, SAVE=False)
    #
    correl_zscan(points_3d_SM3, yaml_file=yaml_file_SM3, images_path=images_path_SM3, Nimg=10, win_size=15,
                 output=os.path.join('output', images_path_SM3.split('/')[-1]), DEBUG=False, SAVE=False)

    correl_zscan(points_3d_SM4, yaml_file=yaml_file_SM4, images_path=images_path_SM4, Nimg=10, win_size=15,
                 output=os.path.join('output', images_path_SM4.split('/')[-1]), DEBUG=False, SAVE=False)


if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
