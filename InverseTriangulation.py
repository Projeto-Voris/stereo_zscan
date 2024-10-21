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
    def __init__(self, yaml_file, gcs3d_pts, left_imgs, right_imgs):
        if len(left_imgs) != len(right_imgs):
            raise Exception("Number of images do not match")
        self.yaml_file = yaml_file
        self.left_imgs = left_imgs
        self.right_imgs = right_imgs

        # Initialize all camera parameters in a single nested dictionary
        self.camera_params = {
            'left_cam': {'kk': np.array([]), 'kc': np.array([]), 'r': np.array([]), 't': np.array([])},
            'right_cam': {'kk': np.array([]), 'kc': np.array([]), 'r': np.array([]), 't': np.array([])},
            'stereo': {'R': np.array([]), 'T': np.array([])}
        }

        self.gcs3d_pts = gcs3d_pts
        self.z_scan_step = np.unique(gcs3d_pts[:, 2]).shape[0]

        self.max_gpu_usage = self.set_datalimit_gpu() // 3

    def read_yaml_file(self):
        """
        Read YAML file to extract cameras parameters
        """
        # Load the YAML file
        with open(self.yaml_file) as file:  # Replace with your file path
            params = yaml.safe_load(file)

            # Parse the matrices
        self.camera_params['left_cam']['kk'] = np.array(params['camera_matrix_left'], dtype=np.float64)
        self.camera_params['left_cam']['kc'] = np.array(params['dist_coeffs_left'], dtype=np.float64)
        self.camera_params['left_cam']['r'] = np.array(params['rot_matrix_left'], dtype=np.float64)
        self.camera_params['left_cam']['t'] = np.array(params['t_left'], dtype=np.float64)

        self.camera_params['right_cam']['kk'] = np.array(params['camera_matrix_right'], dtype=np.float64)
        self.camera_params['right_cam']['kc'] = np.array(params['dist_coeffs_right'], dtype=np.float64)
        self.camera_params['right_cam']['r'] = np.array(params['rot_matrix_right'], dtype=np.float64)
        self.camera_params['right_cam']['t'] = np.array(params['t_right'], dtype=np.float64)

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

    def set_datalimit_gpu(self):
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

    def transform_gcs2ccs_gpu(self, cam_name):
        """
        Transform Global Coordinate System (xg, yg, zg)
         to Camera's Coordinate System (xc, yc, zc) and transform to Image's plane (uv)
         Returns:
             uv_image_points: (N, 2) reprojected points to image's plane
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
            xyz_ccs_norm_dist = self.undistorted_points_gpu(xyz_ccs_norm.T, dist)
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

        return uv_points[:, :2]

    def undistorted_points_gpu(self, points, dist):
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

    def bi_interpolation(self,images, uv_points):
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

    def fringe_masks(self , uv_l, uv_r, std_l, std_r, phi_id, min_thresh=0, max_thresh=1):
        valid_u_l = (uv_l[0, :] >= 0) & (uv_l[0, :] < self.left_imgs.shape[1])
        valid_v_l = (uv_l[1, :] >= 0) & (uv_l[1, :] < self.left_imgs.shape[0])
        valid_u_r = (uv_r[0, :] >= 0) & (uv_r[0, :] < self.right_imgs.shape[1])
        valid_v_r = (uv_r[1, :] >= 0) & (uv_r[1, :] < self.right_imgs.shape[0])
        valid_uv = valid_u_l & valid_u_r & valid_v_l & valid_v_r
        phi_mask = np.zeros(uv_l.shape[1], dtype=bool)
        phi_mask[phi_id] = True
        valid_l = (min_thresh < std_l) & (std_l < max_thresh)
        valid_r = (min_thresh < std_r) & (std_r < max_thresh)

        valid_std = valid_r & valid_l

        return valid_uv & valid_std & phi_mask

    def fringe_process(self):
        t0 = time.time()
        uv_left = self.transform_gcs2ccs_gpu(cam_name='left')
        uv_right = self.transform_gcs2ccs_gpu(cam_name='right')

        inter_left, std_left = self.bi_interpolation(self.left_imgs, uv_left)
        inter_right, std_right = self.bi_interpolation(self.right_imgs, uv_right)

        phi_min_id = self.phase_map(inter_left, inter_right)
        measured_pts = self.gcs3d_pts[self.fringe_masks(uv_left, uv_right, std_left, std_right, phi_min_id)]

        print('Zscan result dt: {} s'.format(round(time.time() - t0),2))
        return measured_pts
