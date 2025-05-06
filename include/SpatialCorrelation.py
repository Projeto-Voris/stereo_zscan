import numpy as np
import cupy as cp
import yaml
import os
import matplotlib.pyplot as plt
import cv2
import gc
from scipy.spatial import cKDTree
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
        rot = cp.asarray(self.camera_params[cam_name]['r'])
        tran = cp.asarray(self.camera_params[cam_name]['t'])

        # Estimate the size of the input and output arrays
        # num_points = xyz_gcs.shape[0]
        bytes_per_float32 = 8  # Simulate double-precision float usage

        # Estimate the memory required per point for transformation and intermediate steps
        memory_per_point = (4 * 3 * bytes_per_float32) + (3 * bytes_per_float32)  # For xyz_gcs_1 and xyz_ccs
        total_memory_required = points_3d.shape[0] * memory_per_point

        # Adjust the batch size based on memory limitations
        if total_memory_required > self.max_gpu_usage * 1024 ** 3:
            points_per_batch = int(
                (self.max_gpu_usage * 1024 ** 3 // memory_per_point) // 10)  # Reduce batch size more aggressively
            # print(f"Processing {points_per_batch} points per batch due to memory limitations.")
        else:
            points_per_batch = points_3d.shape[0] # Process all points at once

        # Initialize an empty list to store results (on the CPU)
        uv_points_list = cp.empty((2, xyz_gcs.shape[0]), dtype=np.float16)

        # Process points in batches
        for i in range(0, points_3d.shape[0], points_per_batch):
            end = min(i + points_per_batch, points_3d.shape[0])
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

        return uv_points_list[:2, :] # (2, N*Nz)

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

        self.grid = cp.asarray(points, dtype=cp.float16)  # shape (Nx, Ny, Nz, 3)

    def bi_interpolation(self, images, uv_points):
        """
        Perform bilinear interpolation on a stack of images at specified uv_points on the GPU.

        Parameters:
        ----------
        images : (height, width, num_images) array or (height, width) for a single image.
        uv_points : (2, N) array of UV points where N is the number of points.

        Returns:
        -------
        interpolated : cp.ndarray
            Interpolated pixel values for each point.
        std : cp.ndarray
            Standard deviation of the corner pixels used for interpolation.
        """
        images = cp.asarray(images)
        uv_points = cp.asarray(uv_points)

        if len(images.shape) == 2:  # Convert single image to a stack with one image
            images = images[:, :, cp.newaxis]

        height, width, num_images = images.shape

        # Estimate memory usage per point
        memory_per_point = 8 * num_images * 4
        points_per_batch = max(1, int(self.max_gpu_usage * 1024 ** 3 // memory_per_point))

        # Output arrays on GPU
        interpolated = cp.zeros((uv_points.shape[1], num_images), dtype=cp.float16)
        std = cp.zeros((uv_points.shape[1], num_images), dtype=cp.float16)

        for i in range(0, uv_points.shape[1], points_per_batch):
            end = min(i + points_per_batch, uv_points.shape[1])
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

    def extract_kernels_batch(self, r_xy, stride):
        """
        Vectorized extraction of all (Kx, Ky, Nz, 3) kernels at once.
        Parameters:
            r_xy   : float     – radius in world units around each (x0,y0)
            stride : float     – step size between kernel centers (world units)
        Returns:
            kernels : cp.ndarray, shape (N_centers, Kx, Ky, Nz, 3)
            centers : list[(float,float)], length N_centers
        """
        # grid shape: (Nx, Ny, Nz, 3)
        Nx, Ny, Nz, _ = self.grid.shape
        dx = float(self.x_vals[1] - self.x_vals[0])
        dy = float(self.y_vals[1] - self.y_vals[0])

        # compute index‐radius
        r_ix = int(round(r_xy / dx))
        r_iy = int(round(r_xy / dy))

        # compute stride in index units (at least 1)
        stride_ix = max(1, int(round(stride / dx)))
        stride_iy = max(1, int(round(stride / dy)))

        # valid center index ranges
        ix_min, ix_max = r_ix, Nx - r_ix
        iy_min, iy_max = r_iy, Ny - r_iy

        # build index arrays for centers
        ix_centers = cp.arange(ix_min, ix_max, stride_ix)  # shape (N_centers_x,)
        iy_centers = cp.arange(iy_min, iy_max, stride_iy)  # shape (N_centers_y,)

        # make a meshgrid of those center‐indices
        IX, IY = cp.meshgrid(ix_centers, iy_centers, indexing='ij')
        IX = IX.ravel()  # (N_centers,)
        IY = IY.ravel()  # (N_centers,)
        N_centers = IX.size

        # record real‐world center coordinates
        centers_x = self.x_vals[IX]  # (N_centers,)
        centers_y = self.y_vals[IY]  # (N_centers,)
        centers = list(zip(cp.asnumpy(centers_x), cp.asnumpy(centers_y)))

        # build relative offsets for the kernel window
        off_x = cp.arange(-r_ix, r_ix + 1)  # (Kx,)
        off_y = cp.arange(-r_iy, r_iy + 1)  # (Ky,)

        # compute absolute indices for each kernel center
        #   x_idx: (N_centers, Kx)
        x_idx = IX[:, None] + off_x[None, :]
        #   y_idx: (N_centers, Ky)
        y_idx = IY[:, None] + off_y[None, :]

        # gather kernels via advanced indexing:
        #   self.grid has shape (Nx, Ny, Nz, 3)
        #   we want kernels of shape (N_centers, Kx, Ky, Nz, 3)
        kernels = self.grid[
                  x_idx[:, :, None],  # (N_centers, Kx, 1)
                  y_idx[:, None, :],  # (N_centers, 1, Ky)
                  :,  # all Nz
                  :  # all 3 coords
                  ]
        # resulting shape: (N_centers, Kx, Ky, Nz, 3)

        return kernels, centers

    def filter_sparse_points(self, xyz, corr, min_neighbors=5, radius=10):
        """
        Remove sparse points from a 3D point cloud based on spatial density.

        Parameters:
        ----------
        xyz : np.ndarray
            3D points of shape (N, 3).
        corr : np.ndarray
            Correlation values of shape (N,).
        min_neighbors : int
            Minimum number of neighbors required to keep a point.
        radius : float
            Radius within which to count neighbors.

        Returns:
        -------
        filtered_xyz : np.ndarray
            Filtered 3D points.
        filtered_corr : np.ndarray
            Correlation values corresponding to the filtered points.
        """
        # Build a KD-tree for fast neighbor search
        tree = cKDTree(xyz)

        # Query the number of neighbors within the radius for each point
        neighbor_counts = tree.query_ball_point(xyz, r=radius)
        neighbor_counts = np.array([len(neighbors) for neighbors in neighbor_counts])

        # Create a mask for points with sufficient neighbors
        dense_mask = neighbor_counts >= min_neighbors

        # Filter points and correlation values
        filtered_xyz = xyz[dense_mask]
        filtered_corr = corr[dense_mask]

        return filtered_xyz, filtered_corr

    def run_batch(self, r_xy=0.1, stride=0.1):

        # Extract kernels and centers
        # kernels: (N_kernels, Kx, Ky, Nz, 3)
        # centers: list of tuples (x, y)
        kernels, centers = self.extract_kernels_batch(r_xy=r_xy, stride=stride)  # (N_kernels, Kx, Ky, Nz, 3)
        #
        N, Kx, Ky, Nz, _ = kernels.shape

        pts_flat = kernels.reshape(N*Nz, 3)

        uv_left = self.transform_gcs2ccs(pts_flat, cam_name='left')
        uv_right = self.transform_gcs2ccs(pts_flat, cam_name='right')

        # uv_left_flat = uv_left.reshape(-1, 2).T  # Shape: (2, N * P)
        # uv_right_flat = uv_right.reshape(-1, 2).T  # Shape: (2, N * P)

        # interp_L = self.bilinear_interp_batch(self.left_images, uv_left)  # (N, P, T)
        # interp_R = self.bilinear_interp_batch(self.right_images, uv_right)

        interp_L, stdL = self.bi_interpolation(self.left_images, uv_left) #(N, Kx, Ky, T)
        interp_R, stdR = self.bi_interpolation(self.right_images, uv_right)

        interp_L = interp_L.reshape(N, Kx, Ky, Nz, -1)
        interp_R = interp_R.reshape(N, Kx, Ky, Nz, -1)

        # Compute std deviation over time for each point
        std_L = cp.std(stdL, axis=1)  # (N, P)
        std_R = cp.std(stdR, axis=1)  # (N, P)


        corr_all = []

        for z in range(Nz):
            l = interp_L[:, :, :, z, :].reshape(N, interp_L.shape[-1])
            r = interp_R[:, :, :, z, :].reshape(N, interp_R.shape[-1])
            # corr = self.correlate_batch(l, r)
            corr = self.temp_cross_correlation(l, r)
            # corr_mean = cp.nanmean(corr, axis=0)
            corr_all.append(corr)

        corr_all = cp.stack(corr_all, axis=1)  # (N, Nz)
        z_best_idx = cp.nanargmax(corr_all, axis=1)
        z_best = self.z_vals[z_best_idx]
        corr_best = cp.nanmax(corr_all, axis=1)

        # Convert list of (x0, y0) to Cupy array: (N, 2)
        centers_cp = cp.asarray(centers, dtype=cp.float32)  # shape: (N, 2)

        # Combine with z_best to form (N, 3)
        xyz = cp.concatenate((centers_cp, z_best[:, None]), axis=1)  # (N, 3)
        # xyz = xyz[texture_mask]
        # corr_best = corr_best[texture_mask]
        return xyz, corr_best, std_L, std_R

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

        # Initialize outputs with the correct data type (float32 for memory efficiency)
        ho = cp.empty(left_Igray.shape[0], dtype=cp.float32)


        # Load only the current batch into the GPU
        batch_left = cp.asarray(left_Igray, dtype=cp.float32)
        batch_right = cp.asarray(right_Igray, dtype=cp.float32)

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
        ho = ho_batch

        # Release memory after processing each batch
        del batch_left, batch_right, left_mean_batch, right_mean_batch, ho_batch
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()


        return ho