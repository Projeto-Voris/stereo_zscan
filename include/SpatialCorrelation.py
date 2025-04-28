import numpy as np
import cupy as cp
import yaml
import os
import matplotlib.pyplot as plt
import cv2
import gc

class StereoSpatialCorrelator:
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

    def transform_gcs2ccs(self, points_3d, cam_name):
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

    def points3d(self, x_lim, y_lim, z_lim, xy_step, z_step):
        """
        Build full 3D points grid (no memory explosion, sliding kernels will select parts).
        """
        x_lin = np.arange(x_lim[0], x_lim[1] + xy_step, xy_step)
        y_lin = np.arange(y_lim[0], y_lim[1] + xy_step, xy_step)
        z_lin = np.arange(z_lim[0], z_lim[1] + z_step, z_step)

        X, Y, Z = np.meshgrid(x_lin, y_lin, z_lin, indexing='ij')
        points = np.stack((X, Y, Z), axis=-1)  # shape (Nx, Ny, Nz, 3)

        self.x_vals = cp.asarray(x_lin)
        self.y_vals = cp.asarray(y_lin)
        self.z_vals = cp.asarray(z_lin)

        self.grid = cp.asarray(points)
        self.num_points = self.grid.shape[0]*self.grid.shape[1]*self.grid.shape[2]
        self.z_scan_step = np.unique(self.grid[:,2]).shape[0]//self.num_points

        return self.grid  # (Nx, Ny, Nz, 3)

    def bilinear_interpolation(self, images, uv_coords):
        """
        Bilinear interpolation of grayscale images at given UV points.
        """
        H, W, T = images.shape
        x = uv_coords[:, 0]
        y = uv_coords[:, 1]

        x0 = cp.floor(x).astype(cp.int32)
        x1 = cp.clip(x0 + 1, 0, W-1)
        y0 = cp.floor(y).astype(cp.int32)
        y1 = cp.clip(y0 + 1, 0, H-1)

        wa = (x1 - x) * (y1 - y)
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)

        Ia = images[y0, x0, :]
        Ib = images[y1, x0, :]
        Ic = images[y0, x1, :]
        Id = images[y1, x1, :]

        interpolated = (wa[:, None] * Ia +
                        wb[:, None] * Ib +
                        wc[:, None] * Ic +
                        wd[:, None] * Id)

        return interpolated  # (N_points, T)

    def spatial_correlation(self, interp_left, interp_right):
        """
        Compute Pearson correlation between two sets of interpolated patches.
        """
        mean_left = cp.mean(interp_left, axis=1, keepdims=True)
        mean_right = cp.mean(interp_right, axis=1, keepdims=True)

        num = cp.sum((interp_left - mean_left) * (interp_right - mean_right), axis=1)
        den = cp.sqrt(cp.sum((interp_left - mean_left)**2, axis=1) *
                      cp.sum((interp_right - mean_right)**2, axis=1))

        corr = num / (den + 1e-8)  # (N_points,)
        return corr

    def slide_and_correlate(self, r_xy, stride):
        """
        Slide a 3D kernel across the XY plane, project points, interpolate, correlate.
        """

        Nx, Ny, Nz, _ = self.grid.shape

        results = []

        # Build center locations
        x_centers = cp.arange(self.x_vals[0]+r_xy, self.x_vals[-1]-r_xy, stride)
        y_centers = cp.arange(self.y_vals[0]+r_xy, self.y_vals[-1]-r_xy, stride)

        for x0 in x_centers:
            for y0 in y_centers:

                # Extract local 3D kernel
                mask_x = cp.abs(self.x_vals - x0) <= r_xy
                mask_y = cp.abs(self.y_vals - y0) <= r_xy

                kernel = self.grid[mask_x, :, :, :]
                kernel = kernel[:, mask_y, :, :]  # shape (Kx, Ky, Nz, 3)
                Kx, Ky, Nz, _ = kernel.shape

                if Kx == 0 or Ky == 0:
                    continue

                kernel_points = kernel.reshape(-1, 3)

                # Project to cameras
                uv_left = self.transform_gcs2ccs(kernel_points,cam_name='left').T
                uv_right = self.transform_gcs2ccs(kernel_points, cam_name='right').T

                # Interpolate
                interp_left = self.bilinear_interpolation(self.left_images, uv_left)
                interp_right = self.bilinear_interpolation(self.right_images, uv_right)

                # Reshape interpolated to (Kx, Ky, Nz, T)
                interp_left = interp_left.reshape(Kx, Ky, Nz, -1)
                interp_right = interp_right.reshape(Kx, Ky, Nz, -1)

                # Correlate over depth
                corr_depth = []

                for z_idx in range(Nz):
                    patch_left = interp_left[:, :, z_idx, :].reshape(-1, interp_left.shape[-1])
                    patch_right = interp_right[:, :, z_idx, :].reshape(-1, interp_right.shape[-1])

                    corr = self.spatial_correlation(patch_left, patch_right)
                    corr_depth.append(corr.mean())

                corr_depth = cp.array(corr_depth)
                best_z_idx = cp.argmax(cp.asarray(corr_depth))

                best_z = self.z_vals[best_z_idx]
                best_corr = corr_depth[best_z_idx]

                # Save (x, y, best_z, best_corr)
                results.append(([float(x0), float(y0), float(best_z)], float(best_corr)))

        return results
