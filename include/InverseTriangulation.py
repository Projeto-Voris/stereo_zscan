import cv2
import numpy as np
import cupy as cp
import yaml
import matplotlib.pyplot as plt
import time
import gc
from cupyx.fallback_mode.fallback import ndarray
import os

class InverseTriangulation:
    def __init__(self, yaml_file):

        self.left_images = cp.array([])
        self.right_images = cp.array([])

        # Initialize all camera parameters in a single nested dictionary
        self.camera_params = {
            'left': {'kk': np.array([]), 'kc': np.array([]), 'r': np.array([]), 't': np.array([])},
            'right': {'kk': np.array([]), 'kc': np.array([]), 'r': np.array([]), 't': np.array([])},
            'stereo': {'R': np.array([]), 'T': np.array([])}
        }
        self.read_yaml_file(yaml_file)


        self.z_scan_step = None
        self.num_points = None
        self.max_gpu_usage = self.set_datalimit() // 3

        # self.uv_left = []
        # self.uv_right = []

    def read_yaml_file(self, yaml_file):
        """
        Read YAML file to extract cameras parameters
        """
        # Load the YAML file
        with open(yaml_file) as file:  # Replace with your file path
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

    def read_images(self, path, images_list, n_imgs):
        """
        Read all images from the specified path and stack them into a single array.
        Parameters:
            path: (string) path to images folder.
            images_list: (list of strings) list of image names.
        Returns:
            images: (height, width, number of images) array of images.
        """

        # Read all images using list comprehension
        images = [cv2.imread(os.path.join(path, str(img_name)), cv2.IMREAD_GRAYSCALE)
                    for img_name in images_list[0:n_imgs]]
        # images = np.stack(images, axis=-1).astype(np.uint8)  # Convert to uint8
                            
        return images

    def convert_images(self, left_imgs, right_imgs, apply_clahe=False, tile=11, climp=5.0, undist=False):
        """
        Convert images to CuPy arrays for GPU processing.
        Optionally apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
        """
        if apply_clahe:
            clahe = cv2.createCLAHE(clipLimit=climp, tileGridSize=(tile, tile))
            if undist:
                left_imgs = [self.remove_img_distortion(clahe.apply(img), 'left') for img in left_imgs]
                right_imgs = [self.remove_img_distortion(clahe.apply(img), 'right') for img in right_imgs]
            else:
                left_imgs = [clahe.apply(img) for img in left_imgs]
                right_imgs = [clahe.apply(img) for img in right_imgs]


        self.left_images = cp.asarray(np.stack(left_imgs, axis=-1)).astype(cp.uint8)
        self.right_images = cp.asarray(np.stack(right_imgs, axis=-1)).astype(cp.uint8)
        return True

    def remove_img_distortion(self, img, camera):
        return cv2.undistort(img, self.camera_params[camera]['kk'], self.camera_params[camera]['kc'])

    def points3d(self, x_lim=(-5, 5), y_lim=(-5, 5), z_lim=(0, 5), xy_step=1.0, z_step=1.0, visualize=False):
        """
            Create a 3D space of combination from linear arrays of X Y Z
            Parameters:
                x_lim: Begin and end of linear space of X
                y_lim: Begin and end of linear space of Y
                z_lim: Begin and end of linear space of Z
                xy_step: Step size between X and Y
                z_step: Step size between Z and X
                visualize: Visualize the 3D space
            Returns:
                cube_points: combination of X Y and Z
            """
        x_lin = np.arange(x_lim[0], x_lim[1], xy_step)
        y_lin = np.arange(y_lim[0], y_lim[1], xy_step)
        z_lin = np.arange(z_lim[0], z_lim[1], z_step)

        mg1, mg2, mg3 = np.meshgrid(x_lin, y_lin, z_lin, indexing='ij')

        c_points = np.stack([mg1, mg2, mg3], axis=-1).reshape(-1, 3)

        if visualize:
            self.plot_3d_points(x=c_points[:, 0], y=c_points[:, 1], z=c_points[:, 2])

        self.num_points = c_points.shape[0]
        self.z_scan_step = np.unique(c_points[:, 2]).shape[0]

        return c_points.astype(np.float16)
    
    def point3d_split(self, x_lin, y_lin, z_lin, visualize=False):
        """
            Create a 3D space of combination from linear arrays of X Y Z
            Parameters:
                x_lim: Begin and end of linear space of X
                y_lim: Begin and end of linear space of Y
                z_lim: Begin and end of linear space of Z
                xy_step: Step size between X and Y
                z_step: Step size between Z and X
                visualize: Visualize the 3D space
            Returns:
                cube_points: combination of X Y and Z
            """
        mg1, mg2, mg3 = np.meshgrid(x_lin, y_lin, z_lin, indexing='ij')

        c_points = np.stack([mg1, mg2, mg3], axis=-1).reshape(-1, 3)

        if visualize:
            self.plot_3d_points(x=c_points[:, 0], y=c_points[:, 1], z=c_points[:, 2])

        self.num_points = c_points.shape[0]
        self.z_scan_step = np.unique(c_points[:, 2]).shape[0]

        return c_points.astype(np.float16)

    def kernel_3d(self, kernel_size=3, zlim=(0,10), dx=1, dy=1, dz=1):
        """
        Create a 3D kernel for correlation
        Parameters:
            kernel_size: Size of the kernel (default is 3).
            dx: Step size in the x direction (default is 1).
            dy: Step size in the y direction (default is 1).
            dz: Step size in the z direction (default is 1).
        Returns:
            kernel: 3D kernel for correlation.
        """
        x = np.arange(-kernel_size // 2, kernel_size // 2 + 1, dx)
        y = np.arange(-kernel_size // 2, kernel_size // 2 + 1, dy)
        z = np.arange(zlim[0], zlim[1], dz)  # Create a range for Z based on zlim

        # Create a 3D grid of points
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

        # Stack the grid points into a single array of shape (N, 3)
        kernel_points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)
        self.num_points = kernel_points.shape[0]
        self.z_scan_step = z.shape[0]
        return kernel_points        

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

    def transform_gcs2ccs(self, points_3d, cam_name, undist=False):
        """
        Transform Global Coordinate System (xg, yg, zg)
         to Camera's Coordinate System (xc, yc, zc) and transform to Image's plane (uv)
         Returns:
             uv_image_points: (2,N) reprojected points to image's plane
        """
        # Convert all inputs to CuPy arrays for GPU computation
        xyz_gcs = cp.asarray(points_3d)
        k = cp.asarray(self.camera_params[cam_name]['kk'])
        dist = cp.asarray(self.camera_params[cam_name]['kc'])
        rot = cp.asarray(self.camera_params[cam_name]['r'])
        tran = cp.asarray(self.camera_params[cam_name]['t'])

        # Estimate the size of the input and output arrays
        # num_points = xyz_gcs.shape[0]
        bytes_per_float32 = 8  # Simulate double-precision float usage

        # Estimate the memory required per point for transformation and intermediate steps
        memory_per_point = (4 * 3 * bytes_per_float32) + (3 * bytes_per_float32)  # For xyz_gcs_1 and xyz_ccs
        total_memory_required = self.num_points * memory_per_point

        # Adjust the batch size based on memory limitations
        if total_memory_required > self.max_gpu_usage * 1024 ** 3:
            points_per_batch = int(
                (self.max_gpu_usage * 1024 ** 3 // memory_per_point) // 10)  # Reduce batch size more aggressively
            # print(f"Processing {points_per_batch} points per batch due to memory limitations.")
        else:
            points_per_batch = self.num_points  # Process all points at once

        # Initialize an empty list to store results (on the CPU)
        uv_points_list = cp.empty((2, xyz_gcs.shape[0]), dtype=np.float16)

        # Process points in batches
        for i in range(0, self.num_points, points_per_batch):
            end = min(i + points_per_batch, self.num_points)
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
            if undist:
                xyz_ccs_norm_dist = self.undistorted_points(xyz_ccs_norm.T, dist)
                del xyz_ccs_norm  # Free memory
                uv_points_batch = cp.dot(k, xyz_ccs_norm_dist.T).astype(cp.float16)
                del xyz_ccs_norm_dist  # Free memory
            else:
                # Compute image points using the intrinsic matrix K
                uv_points_batch = cp.dot(k, xyz_ccs_norm).astype(cp.float16)
                del xyz_ccs_norm  # Free memory

            # Debug: Check the shape of the result
            # print(f"uv_points_batch shape: {uv_points_batch.shape}")

            # Transfer results back to CPU after processing each batch
            uv_points_list[:, i:end] = uv_points_batch[:2, :]

            # Free GPU memory after processing each batch
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()

        return uv_points_list

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
            # y_indices_l = cp.clip(y1_l[:, :, None] + offsets[None, :], 0, height - window_size)
            # x_indices_l = cp.clip(x1_l[:, :, None] + offsets[None, :], 0, width - window_size)
            # y_indices_r = cp.clip(y1_r[:, :, None] + offsets[None, :], 0, height - window_size)
            # x_indices_r = cp.clip(x1_r[:, :, None] + offsets[None, :], 0, width - window_size)
    
    def bi_interpolation(self, images, uv_points, window_size=3):
        """
        Perform bilinear interpolation on a stack of images at specified uv_points on the GPU.

        Parameters:
        ----------
        images : (height, width, num_images) array or (height, width) for a single image.
        uv_points : (2, N) array of UV points where N is the number of points.
        window_size : int
            Unused here but kept for consistency with spatial functions.

        Returns:
        -------
        interpolated_cpu : np.ndarray
            Interpolated pixel values for each point.
        std_cpu : np.ndarray
            Standard deviation of the corner pixels used for interpolation.
        """
        images = cp.asarray(images)
        uv_points = cp.asarray(uv_points)

        if len(images.shape) == 2:  # Convert single image to a stack with one image
            images = images[:, :, cp.newaxis]

        height, width, num_images = images.shape

        # Estimate memory usage per point
        memory_per_point = 4 * num_images * 4
        points_per_batch = max(1, int(self.max_gpu_usage * 1024 ** 3 // memory_per_point))

        # Output arrays on GPU
        interpolated = cp.zeros((self.num_points, num_images), dtype=cp.float16)
        std = cp.zeros((self.num_points, num_images), dtype=cp.float16)

        for i in range(0, self.num_points, points_per_batch):
            end = min(i + points_per_batch, self.num_points)
            uv_batch = uv_points[:, i:end]

            # Compute integer and fractional parts of UV coordinates
            x = uv_batch[0].astype(cp.float16)
            y = uv_batch[1].astype(cp.float16)

            x1 = cp.clip(cp.floor(x).astype(cp.int32), 0, width - 1)
            y1 = cp.clip(cp.floor(y).astype(cp.int32), 0, height - 1)
            x2 = cp.clip(x1 + 1, 0, width - 1)
            y2 = cp.clip(y1 + 1, 0, height - 1)

            x_diff = x - x1
            y_diff = y - y1
            for k in range(num_images):
                # Vectorized extraction of corner pixels
                p11 = images[y1, x1, k]  # Top-left
                p12 = images[y2, x1, k]  # Bottom-left
                p21 = images[y1, x2, k]  # Top-right
                p22 = images[y2, x2, k]  # Bottom-right

                # Bilinear interpolation
                interpolated_batch = (
                        p11 * (1 - x_diff) * (1 - y_diff) +
                        p21 * x_diff * (1 - y_diff) +
                        p12 * (1 - x_diff) * y_diff +
                        p22 * x_diff * y_diff
                )

                std_batch = cp.std(cp.vstack([p11, p12, p21, p22]), axis=0)

                # Store results in GPU arrays
                interpolated[i:end, k] = interpolated_batch
                std[i:end, k] = std_batch

            del p11, p12, p21, p22, std_batch, interpolated_batch
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()
        # Convert results to CPU
        # interpolated_cpu = cp.asnumpy(interpolated_gpu)
        # std_cpu = cp.asnumpy(std_gpu)


        return interpolated, std

    def spatial_correl(self, uv_left, uv_right, window_size=3, save_points=False, name_file='reshaped_corr'):
        """
        Compute spatial correlation for patches around specified points across all images.

        Parameters:
        ----------
        uv_left : (2, N) np.ndarray
            UV coordinates for points in the left image.
        uv_right : (2, N) np.ndarray
            UV coordinates for points in the right image.
        window_size : int
            Size of the patch window around each point (default is 3).

        Returns:
        -------
        spatial_corr : np.ndarray
            Spatial correlation values for each point.
        spatial_max : np.ndarray
            Maximum correlation value for each spatial window.
        spatial_id : np.ndarray
            Index of the maximum correlation value for each point.
        """

        half_window = window_size // 2
        height, width, num_images = self.left_images.shape

        # Estimate memory usage and adjust batch size
        memory_per_point = 2 * window_size * window_size * num_images * 4  # Approx memory per point
        points_per_batch = max(1, int(self.max_gpu_usage * 1024 ** 3 // memory_per_point))
        # points_per_batch = 10
        # Allocate space for results
        spatial_corr = cp.zeros((self.num_points), dtype=cp.float16)
        std_corr = cp.zeros((self.num_points), dtype=cp.float16)

        for i in range(0, self.num_points, points_per_batch):
            end_idx = min(i + points_per_batch, self.num_points)  # or np.int64
            # Process the batch (mean, std, correlation computation, etc.)
            uv_batch_l = cp.asarray(uv_left[:, i:end_idx])
            uv_batch_r = cp.asarray(uv_right[:, i:end_idx])

            # Compute integer and fractional parts of UV coordinates
            x1_l = cp.clip(cp.floor(uv_batch_l[0] - half_window).astype(cp.int32), 0, width - window_size)
            y1_l = cp.clip(cp.floor(uv_batch_l[1] - half_window).astype(cp.int32), 0, height - window_size)

            x1_r = cp.clip(cp.floor(uv_batch_r[0] - half_window).astype(cp.int32), 0, width - window_size)
            y1_r = cp.clip(cp.floor(uv_batch_r[1] - half_window).astype(cp.int32), 0, height - window_size)

            offsets = cp.arange(window_size)

            # Compute indices for batch-based patch extraction
            y_indices_l = cp.clip(y1_l[:, None] + offsets[None, :], 0, height - window_size)
            x_indices_l = cp.clip(x1_l[:, None] + offsets[None, :], 0, width - window_size)
            y_indices_r = cp.clip(y1_r[:, None] + offsets[None, :], 0, height - window_size)
            x_indices_r = cp.clip(x1_r[:, None] + offsets[None, :], 0, width - window_size)

            # Extract patches for left and right images
            patch_left = self.left_images[y_indices_l[:, :, None], x_indices_l[:, None, :], :].astype(cp.float32)
            patch_right = self.right_images[y_indices_r[:, :, None], x_indices_r[:, None, :], :].astype(cp.float32)

            # Compute mean subtraction
            mean_left = cp.mean(patch_left, axis=(1, 2), keepdims=True)
            std_left = cp.std(patch_left, axis=(1, 2))
            mean_right = cp.mean(patch_right, axis=(1, 2), keepdims=True)
            std_right = cp.std(patch_right, axis=(1, 2))

            comb_std_left = cp.mean(std_left, axis=1).astype(cp.float16)
            comb_std_right = cp.mean(std_right, axis=1).astype(cp.float16)

            patch_left -= mean_left.astype(cp.float16)
            patch_right -= mean_right.astype(cp.float16)

            # Compute spatial correlation
            num = cp.sum(patch_left * patch_right, axis=(1, 2))
            den_left = cp.sum(patch_left ** 2, axis=(1, 2))
            den_right = cp.sum(patch_right ** 2, axis=(1, 2))

            # Ensure that comb_std_left and comb_std_right match the batch size exactly
            std_corr[i:end_idx] = cp.sqrt(comb_std_left ** 2 + comb_std_right ** 2)

            spatial_corr[i:end_idx] = cp.sum(num, axis=1) / (
                    cp.sqrt(cp.sum(den_left, axis=1) * cp.sum(den_right, axis=1)) + 1e-10
            )

            # Clear variables to manage memory
            del uv_batch_r, uv_batch_l, patch_left, patch_right, num, den_left, den_right
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()

        # Reshape and compute maximum correlations
        reshaped_corr = spatial_corr.reshape((-1, self.z_scan_step))
        spatial_max = cp.nanmax(reshaped_corr, axis=1)
        spatial_id = cp.nanargmax(reshaped_corr, axis=1) + cp.arange(reshaped_corr.shape[0]) * self.z_scan_step
        std_corr = std_corr[spatial_id]

        # Save reshaped correlation as numpy vector to a text file
        if save_points:
            reshaped_corr_cpu = cp.asnumpy(reshaped_corr)
            np.savetxt('{}.txt'.format(name_file), reshaped_corr_cpu.flatten(), fmt='%.6f')

        # Return as Cupy array
        return spatial_id, spatial_max, std_corr

    def correl_mask(self, std_correl, correl_max, std_thresh, correl_thresh):
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

        correl_mask = cp.zeros(correl_max.shape[0], dtype=bool)
        std_mask = cp.zeros(std_correl.shape[0], dtype=bool)
        correl_mask[correl_max > correl_thresh] = True
        std_mask[std_correl > std_thresh] = True

        return correl_mask & std_mask

    def temp_cross_correlation(self, left_Igray, right_Igray):
        """
        Calculate the cross-correlation between two sets of images over time using CuPy for GPU acceleration,
        while limiting GPU memory usage and handling variable batch sizes.

        Parameters:
        left_Igray: (num_points, num_images) array of left images in grayscale.
        right_Igray: (num_points, num_images) array of right images in grayscale.
        Returns:
        ho: Cross-correlation values.
        Imax: Indices of maximum correlation values.
        """

        num_images = left_Igray.shape[1]  # Number of images

        # Number of tested Z (this will be used as batch size)
        num_pairs = self.num_points // self.z_scan_step  # Total number of XY points

        # Estimate memory usage per point
        bytes_per_float32 = 8
        memory_per_point = (4 * num_images * bytes_per_float32)
        # For left_Igray, right_Igray, and intermediate calculations
        total_memory_required = self.num_points * memory_per_point

        # Adjust the batch size based on memory limitations
        if total_memory_required > self.max_gpu_usage * 1024 ** 3:
            points_per_batch = int(self.max_gpu_usage * 1024 ** 3 // memory_per_point // 10)
            # print(f"Processing {points_per_batch} points per batch due to memory limitations.")
        else:
            points_per_batch = self.num_points  # Process all points at once

        # Initialize outputs with the correct data type (float32 for memory efficiency)
        ho = cp.empty(left_Igray.shape[0], dtype=cp.float32)


        # Process images in chunks based on the adjusted points_per_batch size
        for i in range(0, self.num_points, points_per_batch):
            end = min(i + points_per_batch, self.num_points)

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


        return ho

    def correlation_process(self, points_3d, win_size=3, threshold=0.8):
        """
        Zscan process of temporal and spatial correlations.
        Parameters:
            points_3d: ndarray(N,3) xyz meshgrid of points
            win_size: (int) spatial window size
            correl_param: tuple(float) fused correlation parameter (spatial, correlation)
            threshold: (float) threshold of correlation coefficient
            save_points: (bool) whether to save temporal and spatial correlation images or not.
            visualize: (bool) whether to visualize correlation images or not.
        Returns:
            correl_points: (X*Y, 3) list from bests correlation points.
        """
        uv_left = self.transform_gcs2ccs(points_3d=points_3d, cam_name='left')
        uv_right = self.transform_gcs2ccs(points_3d=points_3d, cam_name='right')

       
        spatial_id, spatial_max, std_corr = self.spatial_correl(window_size=win_size, uv_left=uv_left,
                                                                uv_right=uv_right)
        correl_mask = self.correl_mask(std_correl=std_corr, correl_max=spatial_max, correl_thresh=threshold,
                                       std_thresh=60)
        space_temp_correl_pt = points_3d[np.asarray(cp.asnumpy(spatial_id[correl_mask])).astype(np.int32)]

       
        del uv_left, uv_right, spatial_id, spatial_max, std_corr
        return space_temp_correl_pt
  
    def save_points(self, points, filename, delimiter=','):
        """
        Save 3D points to a text file.
        Parameters:
            points: (N, 3) array of 3D points.
        """
        np.savetxt(filename, points, fmt='%.6f', delimiter=delimiter)
        return True
    
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
