import numpy as np
import cupy as cp
import cv2
import yaml
import matplotlib.pyplot as plt
import os
import time
import gc
from cupyx.fallback_mode import numpy


class InverseTriangulation:
    def __init__(self, yaml_file, gcs3d_pts):

        self.yaml_file = yaml_file
        self.left_images = []
        self.right_images = []

        # Initialize all camera parameters in a single nested dictionary
        self.camera_params = {
            'left': {'kk': np.array([]), 'kc': np.array([]), 'r': np.array([]), 't': np.array([])},
            'right': {'kk': np.array([]), 'kc': np.array([]), 'r': np.array([]), 't': np.array([])},
            'stereo': {'R': np.array([]), 'T': np.array([])}
        }

        self.read_yaml_file()

        self.gcs3d_pts = gcs3d_pts
        self.z_scan_step = np.unique(gcs3d_pts[:, 2]).shape[0]
        self.max_gpu_usage = self.set_datalimit() // 3

        self.uv_left = self.transform_gcs2ccs(cam_name='left')
        self.uv_right = self.transform_gcs2ccs(cam_name='right')

    def read_images(self, left_imgs, right_imgs):
        if len(left_imgs) != len(right_imgs):
            raise Exception("Number of images do not match")
        self.left_images = left_imgs
        self.right_images = right_imgs


    def plot_3d_points(self, x, y, z, color=None, title='Plot 3D of max correlation points'):
        """
        Plot 3D points as scatter points where color is based on Z value
        Parameters:
            x: array of x positions
            y: array of y positions
            z: array of z positions
            color: Vector of point intensity grayscale
        """
        if color is None:
            color = z
        cmap = 'viridis'
        # Plot the 3D scatter plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.title.set_text(title)

        scatter = ax.scatter(x, y, z, c=color, cmap=cmap, marker='o')
        # ax.set_zlim(0, np.max(z))
        colorbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
        colorbar.set_label('Z Value Gradient')

        # Add labels
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel('Z [mm]')
        ax.set_aspect('equal', adjustable='box')
        plt.show()

    def read_yaml_file(self):
        """
        Read YAML file to extract cameras parameters
        """
        # Load the YAML file
        with open(self.yaml_file) as file:  # Replace with your file path
            params = yaml.safe_load(file)

            # Parse the matrices
        self.camera_params['left']['kk'] = np.array(params['camera_matrix_left'], dtype=np.float64)
        self.camera_params['left']['kc'] = np.array(params['dist_coeffs_left'], dtype=np.float64)
        self.camera_params['left']['r'] = np.array(params['rot_matrix_left'], dtype=np.float64)
        self.camera_params['left']['t'] = np.array(params['t_left'], dtype=np.float64)

        self.camera_params['right']['kk'] = np.array(params['camera_matrix_right'], dtype=np.float64)
        self.camera_params['right']['kc'] = np.array(params['dist_coeffs_right'], dtype=np.float64)
        self.camera_params['right']['r'] = np.array(params['rot_matrix_right'], dtype=np.float64)
        self.camera_params['right']['t'] = np.array(params['t_right'], dtype=np.float64)

        self.camera_params['stereo']['R'] = np.array(params['R'], dtype=np.float64)
        self.camera_params['stereo']['T'] = np.array(params['T'], dtype=np.float64)

    def save_points(self, data, filename, delimiter=','):
        """
        Save a 2D NumPy array to a CSV file.

        :param array: 2D numpy array
        :param filename: Output CSV filename
        """
        # Save the 2D array as a CSV file
        np.savetxt(filename, data, delimiter=delimiter)
        print(f"Array saved to {filename}")

    def set_datalimit(self):
        """
        Identify gpu limit
        """
        # Create a device object for the first GPU (device ID 0)
        device_id = 0
        cp.cuda.Device(device_id).use()  # Set the current device
        # Get the total memory in bytes using runtime API
        total_memory = cp.cuda.runtime.getDeviceProperties(device_id)['totalGlobalMem']
        # Convert bytes to GB
        return total_memory / (1024 ** 3)

    def transform_gcs2ccs(self, cam_name):
        """
        Transform Global Coordinate System (xg, yg, zg)
         to Camera's Coordinate System (xc, yc, zc) and transform to Image's plane (uv)
         Returns:
             uv_image_points: (2,N) reprojected points to image's plane
        """
        # Convert all inputs to CuPy arrays for GPU computation
        xyz_gcs = cp.asarray(self.gcs3d_pts)
        k = cp.asarray(self.camera_params[cam_name]['kk'])
        dist = cp.asarray(self.camera_params[cam_name]['kc'])
        rot = cp.asarray(self.camera_params[cam_name]['r'])
        tran = cp.asarray(self.camera_params[cam_name]['t'])

        # Estimate the size of the input and output arrays
        num_points = xyz_gcs.shape[0]
        bytes_per_float32 = 8  # Simulate double-precision float usage

        # Estimate the memory required per point for transformation and intermediate steps
        memory_per_point = (4 * 3 * bytes_per_float32) + (3 * bytes_per_float32)  # For xyz_gcs_1 and xyz_ccs
        total_memory_required = num_points * memory_per_point

        # Adjust the batch size based on memory limitations
        if total_memory_required > self.max_gpu_usage * 1024 ** 3:
            points_per_batch = int(
                (self.max_gpu_usage * 1024 ** 3 // memory_per_point) // 10)  # Reduce batch size more aggressively
            # print(f"Processing {points_per_batch} points per batch due to memory limitations.")
        else:
            points_per_batch = num_points  # Process all points at once

        # Initialize an empty list to store results (on the CPU)
        uv_points_list = []

        # Process points in batches
        for i in range(0, num_points, points_per_batch):
            end = min(i + points_per_batch, num_points)
            xyz_gcs_batch = xyz_gcs[i:end]

            # Debug: Check the shape of the batch
            # print(f"Processing batch {i // points_per_batch + 1}, size: {xyz_gcs_batch.shape}")

            # Add one extra line of ones to the global coordinates
            ones = cp.ones((xyz_gcs_batch.shape[0], 1), dtype=cp.float16)  # Double-precision floats
            xyz_gcs_1 = cp.hstack((xyz_gcs_batch, ones))

            # Create the rotation and translation matrix
            rt_matrix = cp.vstack(
                (cp.hstack((rot, tran[:, None])), cp.array([0, 0, 0, 1], dtype=cp.float16))
            )

            # Multiply the RT matrix with global points [X; Y; Z; 1]
            xyz_ccs = cp.dot(rt_matrix, xyz_gcs_1.T)
            del xyz_gcs_1  # Immediately delete

            # Normalize by dividing by Z to get normalized image coordinates
            epsilon = 1e-10  # Small value to prevent division by zero
            xyz_ccs_norm = cp.hstack(
                (xyz_ccs[:2, :].T / cp.maximum(xyz_ccs[2, :, cp.newaxis], epsilon),
                 cp.ones((xyz_ccs.shape[1], 1), dtype=cp.float16))
            ).T
            del xyz_ccs  # Immediately delete

            # Apply distortion using the GPU
            xyz_ccs_norm_dist = self.undistorted_points(xyz_ccs_norm.T, dist)
            del xyz_ccs_norm  # Free memory

            # Compute image points using the intrinsic matrix K
            uv_points_batch = cp.dot(k, xyz_ccs_norm_dist.T)
            del xyz_ccs_norm_dist  # Free memory

            # Debug: Check the shape of the result
            # print(f"uv_points_batch shape: {uv_points_batch.shape}")

            # Transfer results back to CPU after processing each batch
            uv_points_list.append(cp.asnumpy(uv_points_batch))

            # Free GPU memory after processing each batch
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()

        # Ensure consistent dimensions when concatenating batches
        try:
            # Concatenate all batches along axis 0 (rows)
            uv_points = np.hstack(uv_points_list)  # Use np.hstack for matching shapes

        except ValueError as e:
            print(f"Error during concatenation: {e}")
            raise

        return uv_points[:2, :]

    def undistorted_points(self, points, dist):
        """
        GPU version of the undistorted points function using CuPy.
        Applies radial and tangential distortion.

        Parameters:
            points: 2D array of normalized image coordinates [x, y].
            dist: Distortion coefficients [k1, k2, p1, p2, k3].

        Returns:
            Distorted points on the GPU.
        """
        # Extract distortion coefficients
        k1, k2, p1, p2, k3 = dist

        # Split points into x and y coordinates
        x, y = points[:, 0], points[:, 1]

        # Calculate r^2 (squared distance from the origin)
        r2 = x ** 2 + y ** 2

        # Radial distortion
        radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3

        # Tangential distortion
        x_tangential = 2 * p1 * x * y + p2 * (r2 + 2 * x ** 2)
        y_tangential = p1 * (r2 + 2 * y ** 2) + 2 * p2 * x * y

        # Compute distorted coordinates
        x_distorted = x * radial + x_tangential
        y_distorted = y * radial + y_tangential

        # Stack the distorted points
        distorted_points = cp.vstack([x_distorted, y_distorted, cp.ones_like(x)]).T

        # Clean up intermediate variables to free memory
        del x, y, r2, radial, x_tangential, y_tangential
        cp.get_default_memory_pool().free_all_blocks()

        return distorted_points

    def bi_interpolation(self, images, uv_points):
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

        # Adjust the batch size based on memory limitations
        if total_memory_required > self.max_gpu_usage * 1024 ** 3:
            points_per_batch = int(self.max_gpu_usage * 1024 ** 3 // memory_per_point // 100)
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

    def phase_map(self, interp_left, interp_right, debug=False):
        """
        Identify minimum phase map value
        Parameters:
            interp_left: left interpolated points
            interp_right: right interpolated points
            debug: if true, visualize phi_map array
        Returns:
            phi_min_id: indices of minimum phase map values.
        """
        phi_map = []
        phi_min_id = []
        for k in range(self.gcs3d_pts.shape[0] // self.z_scan_step):
            diff_phi = np.abs(interp_left[self.z_scan_step * k:(k + 1) * self.z_scan_step]
                              - interp_right[self.z_scan_step * k:(k + 1) * self.z_scan_step])
            phi_min_id.append(np.argmin(diff_phi) + k * self.z_scan_step)
            if debug:
                phi_map.append(diff_phi)
        if debug:
            plt.figure()
            plt.plot(phi_map)
            plt.show()
        return phi_min_id

    def fringe_masks(self, std_l, std_r, phi_id, min_thresh=0, max_thresh=1):
        """
        Mask from fringe process to remove outbounds points.
        Paramenters:
            std_l: STD interpolation image's points
            std_r: STD interpolation image's points
            phi_id: Indices for min phase difference
            min_thresh: max threshold for STD
            max_thresh: min threshold for STD
        Returns:
             valid_mask: Valid 3D points on image's plane
        """
        valid_u_l = (self.uv_left[0, :] >= 0) & (self.uv_left[0, :] < self.left_images.shape[1])
        valid_v_l = (self.uv_left[1, :] >= 0) & (self.uv_left[1, :] < self.left_images.shape[0])
        valid_u_r = (self.uv_right[0, :] >= 0) & (self.uv_right[0, :] < self.right_images.shape[1])
        valid_v_r = (self.uv_right[1, :] >= 0) & (self.uv_right[1, :] < self.right_images.shape[0])
        valid_uv = valid_u_l & valid_u_r & valid_v_l & valid_v_r
        phi_mask = np.zeros(self.uv_left.shape[1], dtype=bool)
        phi_mask[phi_id] = True
        valid_l = (min_thresh < std_l) & (std_l < max_thresh)
        valid_r = (min_thresh < std_r) & (std_r < max_thresh)

        valid_std = valid_r & valid_l

        return valid_uv & valid_std & phi_mask

    def fringe_process(self, save_points=True, visualize=False):
        """
        Zscan for stereo fringe process
        Parameters:
            save_points: boolean to save or not image
            visualize: boolean to visualize result
        :return:
        """
        t0 = time.time()

        inter_left, std_left = self.bi_interpolation(self.left_images, self.uv_left)
        inter_right, std_right = self.bi_interpolation(self.right_images, self.uv_right)

        phi_min_id = self.phase_map(inter_left, inter_right)
        measured_pts = self.gcs3d_pts[self.fringe_masks(std_left, std_right, phi_min_id)]

        print('Zscan result dt: {} s'.format(round(time.time() - t0), 2))

        if save_points:
            self.save_points(measured_pts, filename='./output_points.csv')

        if visualize:
            self.plot_3d_points(measured_pts[:, 0], measured_pts[:, 1], measured_pts[:, 2], color=None,
                                title="Fringe process output points")

        return measured_pts

    def temp_cross_correlation(self, left_Igray, right_Igray):
        """
        Calculate the cross-correlation between two sets of images over time using CuPy for GPU acceleration,
        while limiting GPU memory usage and handling variable batch sizes.

        Parameters:
        left_Igray: (num_points, num_images) array of left images in grayscale.
        right_Igray: (num_points, num_images) array of right images in grayscale.
        Returns:
        ho: Cross-correlation values.
        hmax: Maximum correlation values.
        Imax: Indices of maximum correlation values.
        ho_ztep: List of correlation values for all Z value of each XY.
        """

        num_images = left_Igray.shape[1]  # Number of images
        num_points = left_Igray.shape[0]  # Number of points

        # Number of tested Z (this will be used as batch size)
        num_pairs = self.gcs3d_pts.shape[0] // self.z_scan_step  # Total number of XY points

        # Estimate memory usage per point
        bytes_per_float32 = 8
        memory_per_point = (4 * num_images * bytes_per_float32)
        # For left_Igray, right_Igray, and intermediate calculations
        total_memory_required = num_points * memory_per_point

        # Adjust the batch size based on memory limitations
        if total_memory_required > self.max_gpu_usage * 1024 ** 3:
            points_per_batch = int(self.max_gpu_usage * 1024 ** 3 // memory_per_point // 10)
            # print(f"Processing {points_per_batch} points per batch due to memory limitations.")
        else:
            points_per_batch = num_points  # Process all points at once

        # Initialize outputs with the correct data type (float32 for memory efficiency)
        ho = cp.empty(num_points, dtype=cp.float32)

        # Preallocate ho_ztep only if necessary
        ho_ztep = cp.empty((self.z_scan_step, num_pairs),
                           dtype=cp.float32)  # Store values for each Z value tested per XY pair

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
        for k in range(self.gcs3d_pts.shape[0] // self.z_scan_step):
            start_idx = k * self.z_scan_step
            end_idx = (k + 1) * self.z_scan_step
            ho_range = ho[start_idx:end_idx]

            hmax[k] = cp.nanmax(ho_range)
            Imax[k] = cp.nanargmax(ho_range) + k * self.z_scan_step

        return cp.asnumpy(ho), cp.asnumpy(hmax), cp.asnumpy(Imax), cp.asnumpy(ho_ztep)

    def spatial_correl(self, window_size=3):
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
        # If window size <= 5 will be processed on CPU
        if window_size <= 5:
            half_window = window_size // 2
            height, width, num_images = self.left_images.shape
            num_points = self.uv_left.shape[1]

            # Allocate space for the result
            spatial_corr = np.zeros(num_points, dtype=np.float32)

            # Vectorized window bounds for left and right points
            x1_l = np.clip(self.uv_left[0] - half_window, 0, width - window_size).astype(np.int32)
            y1_l = np.clip(self.uv_left[1] - half_window, 0, height - window_size).astype(np.int32)

            x1_r = np.clip(self.uv_right[0] - half_window, 0, width - window_size).astype(np.int32)
            y1_r = np.clip(self.uv_right[1] - half_window, 0, height - window_size).astype(np.int32)

            # Constructing indices for batch-based patch extraction
            # Left patches: Shape (batch_size, window_size, window_size, num_images)
            patch_left = np.array([self.left_images[y:y + window_size, x:x + window_size, :]
                                   for y, x in zip(y1_l, x1_l)], dtype=np.float32)

            # Right patches: Shape (batch_size, window_size, window_size, num_images)
            patch_right = np.array([self.right_images[y:y + window_size, x:x + window_size, :]
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
            spatial_max = np.empty(self.gcs3d_pts.shape[0] // self.z_scan_step, dtype=np.float32)
            spatial_id = np.empty(self.gcs3d_pts.shape[0] // self.z_scan_step, dtype=np.int32)

            for k in range(self.gcs3d_pts.shape[0] // self.z_scan_step):
                spatial_range = spatial_corr[k * self.z_scan_step:(k + 1) * self.z_scan_step]
                spatial_max[k] = np.nanmax(spatial_range)
                spatial_id[k] = np.nanargmax(spatial_range) + k * self.z_scan_step

            return spatial_corr, spatial_max, spatial_id
        if window_size > 5:
            half_window = window_size // 2
            height, width, num_images = self.left_images.shape
            num_points = self.uv_left.shape[1]

            # Estimate memory usage per point
            bytes_per_float32 = 4
            memory_per_point = (
                    2 * window_size * window_size * num_images * bytes_per_float32)  # For left and right patches
            total_memory_required = num_points * memory_per_point

            # Adjust the batch size based on memory limitations
            points_per_batch = max(1, int(self.max_gpu_usage * 1024 ** 3 // memory_per_point))

            # Allocate space for the result on GPU
            spatial_corr = cp.zeros(num_points, dtype=cp.float32)

            for i in range(0, num_points, points_per_batch):
                end_idx = min(i + points_per_batch, num_points)
                uv_batch_l = cp.asarray(self.uv_left[:, i:end_idx])
                uv_batch_r = cp.asarray(self.uv_right[:, i:end_idx])

                # Vectorized window bounds for left and right points
                x1_l = cp.clip(uv_batch_l[0] - half_window, 0, width - window_size).astype(cp.int32)
                y1_l = cp.clip(uv_batch_l[1] - half_window, 0, height - window_size).astype(cp.int32)

                x1_r = cp.clip(uv_batch_r[0] - half_window, 0, width - window_size).astype(cp.int32)
                y1_r = cp.clip(uv_batch_r[1] - half_window, 0, height - window_size).astype(cp.int32)

                # Extract patches for left and right images across all images in the batch
                patch_left = cp.array([self.left_images[y:y + window_size, x:x + window_size, :]
                                       for y, x in zip(y1_l, x1_l)], dtype=cp.float32)

                patch_right = cp.array([self.right_images[y:y + window_size, x:x + window_size, :]
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
            reshaped_corr = spatial_corr.reshape((-1, self.z_scan_step))
            spatial_max = cp.nanmax(reshaped_corr, axis=1)
            spatial_id = cp.nanargmax(reshaped_corr, axis=1) + cp.arange(reshaped_corr.shape[0]) * self.z_scan_step

            # Return results as NumPy arrays
            return cp.asnumpy(spatial_corr), cp.asnumpy(spatial_max), cp.asnumpy(spatial_id)

    def fuse_correlations(self, spatial_corr, temporal_corr, alpha=0.5, beta=0.5):
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

        correl_max = np.empty((self.gcs3d_pts.shape[0] // self.z_scan_step), dtype=np.float32)
        correl_id = np.empty((self.gcs3d_pts.shape[0] // self.z_scan_step), dtype=np.float32)
        for k in range(self.gcs3d_pts.shape[0] // self.z_scan_step):
            correl_range = fused_corr[k * self.z_scan_step:(k + 1) * self.z_scan_step]
            correl_max[k] = np.nanmax(correl_range)
            correl_id[k] = np.nanargmax(correl_range) + k * self.z_scan_step

        return fused_corr, correl_max, correl_id

    def correl_mask(self, std_l, std_r, hmax, min_thresh=0, max_thresh=1):
        """
        Mask from correlation process to remove outbounds points.
        Paramenters:
            std_l: STD interpolation image's points
            std_r: STD interpolation image's points
            phi_id: Indices for min phase difference
            min_thresh: max threshold for STD
            max_thresh: min threshold for STD
        Returns:
             valid_mask: Valid 3D points on image's plane
        """
        valid_u_l = (self.uv_left[0, :] >= 0) & (self.uv_left[0, :] < self.left_images.shape[1])
        valid_v_l = (self.uv_left[1, :] >= 0) & (self.uv_left[1, :] < self.left_images.shape[0])
        valid_u_r = (self.uv_right[0, :] >= 0) & (self.uv_right[0, :] < self.right_images.shape[1])
        valid_v_r = (self.uv_right[1, :] >= 0) & (self.uv_right[1, :] < self.right_images.shape[0])
        valid_uv = valid_u_l & valid_u_r & valid_v_l & valid_v_r

        ho_mask = np.zeros(self.uv_left.shape[1], dtype=bool)
        ho_mask[np.asarray(hmax > 0.95, np.int32)] = True
        if len(std_l.shape) > 1 or len(std_r.shape) > 1:
            std_l = np.std(std_l, axis=1)
            std_r = np.std(std_r, axis=1)

        valid_l = (min_thresh < std_l) & (std_l < max_thresh)
        valid_r = (min_thresh < std_r) & (std_r < max_thresh)

        valid_std = valid_r & valid_l

        return valid_uv & valid_std & ho_mask

    def correlation_process(self, win_size=3, correl_param=(0.3, 0.7), save_points=True, visualize=False):
        """
        Zscan process of temporal and spatial correlations.
        Parameters:
            win_size: (int) spatial window size
            correl_param: tuple(float) fused correlation parameter (spatial, correlation)
            save_points: (bool) whether to save temporal and spatial correlation images or not.
            visualize: (bool) whether to visualize correlation images or not.
        Returns:
            correl_points: (X*Y, 3) list from bests correlation points.
        """
        t0 = time.time()

        inter_left, std_left = self.bi_interpolation(self.left_images, self.uv_left)
        inter_right, std_right = self.bi_interpolation(self.right_images, self.uv_right)

        spatial_corr, spatial_max, spatial_id = self.spatial_correl(window_size=win_size)
        ho, hmax, imax, ho_zstep = self.temp_cross_correlation(left_Igray=inter_left, right_Igray=inter_right)

        fused_corr, fused_max, fused_id = self.fuse_correlations(spatial_corr=spatial_corr, temporal_corr=ho,
                                                                 alpha=correl_param[0], beta=correl_param[1])
        # correl_points = self.gcs3d_pts[self.correl_mask(std_left, std_right, fused_max)]
        correl_points = self.gcs3d_pts[np.asarray(fused_id[fused_max > 0.95]).astype(np.int32)]

        print('Time to process correlation: {} s'.format(round(time.time() - t0), 2))

        if save_points:
            self.save_points(correl_points, filename='./correlation_points.csv')
        if visualize:
            self.plot_3d_points(correl_points[:, 0], correl_points[:, 1], correl_points[:, 2],
                                color=None,
                                title="Fused correlation result")  # fused_max[fused_max > 0.95].astype(np.float32)

        return correl_points
