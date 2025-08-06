import numpy as np
import cupy as cp
import yaml
import os
import matplotlib.pyplot as plt
import cv2
import gc
from sklearn.cluster import DBSCAN

class StereoTemporalSpatialCorrel:
    def __init__(self, yaml_file):

        self.left_images = cp.array([])
        self.right_images = cp.array([])

        self.camera_params = {
            'left': {'kk': cp.array([]), 'kc': cp.array([]), 'r': cp.array([]), 't': cp.array([])},
            'right': {'kk': cp.array([]), 'kc': cp.array([]), 'r': cp.array([]), 't': cp.array([])},
            'stereo': {'R': cp.array([]), 'T': cp.array([])}
        }
        self.read_yaml_file(yaml_file)

        self.max_gpu_usage_gb = self.set_datalimit() 

        self.x_vals = cp.array([])
        self.y_vals = cp.array([])
        self.z_vals = cp.array([])
        self.grid = cp.array([])

    def plot_3d_points(self, x, y, z, color=None, title='Plot 3D of max correlation points'):
        """
        Plot 3D points as scatter points where color is based on Z value
        Parameters:
            x: array of x positions
            y: array of y positions
            z: array of z positions
            color: Vector of point intensity grayscale
        """
        x_cpu = cp.asnumpy(x) if isinstance(x, cp.ndarray) else x
        y_cpu = cp.asnumpy(y) if isinstance(y, cp.ndarray) else y
        z_cpu = cp.asnumpy(z) if isinstance(z, cp.ndarray) else z
        color_cpu = cp.asnumpy(color) if isinstance(color, cp.ndarray) else color

        if color_cpu is None:
            color_cpu = z_cpu
        cmap = 'viridis'
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.title.set_text(title)

        scatter = ax.scatter(x_cpu, y_cpu, z_cpu, c=color_cpu, cmap=cmap, marker='o')
        colorbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
        colorbar.set_label('Z Value Gradient' if color is z else 'Correlation/Intensity')

        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel('Z [mm]')
        ax.set_aspect('equal', adjustable='box')
        plt.show()
    
    def read_yaml_file(self, yaml_file):
        """
        Read YAML file to extract cameras parameters and convert to cp.float32
        """
        with open(yaml_file) as file:
            params = yaml.safe_load(file)

        self.camera_params['left']['kk'] = cp.array(params['camera_matrix_left'], dtype=cp.float32)
        self.camera_params['left']['kc'] = cp.array(params['dist_coeffs_left'], dtype=cp.float32)
        self.camera_params['left']['r'] = cp.array(params['rot_matrix_left'], dtype=cp.float32)
        self.camera_params['left']['t'] = cp.array(params['t_left'], dtype=cp.float32).reshape(3, 1)

        self.camera_params['right']['kk'] = cp.array(params['camera_matrix_right'], dtype=cp.float32)
        self.camera_params['right']['kc'] = cp.array(params['dist_coeffs_right'], dtype=cp.float32)
        self.camera_params['right']['r'] = cp.array(params['rot_matrix_right'], dtype=cp.float32)
        self.camera_params['right']['t'] = cp.array(params['t_right'], dtype=cp.float32).reshape(3, 1)

        self.camera_params['stereo']['R'] = cp.array(params['R'], dtype=cp.float32)
        self.camera_params['stereo']['T'] = cp.array(params['T'], dtype=cp.float32).reshape(3, 1)


    def read_images(self, path, images_list, n_imgs):
        images_cpu = [cv2.imread(os.path.join(path, str(img_name)), cv2.IMREAD_GRAYSCALE)
                      for img_name in images_list[0:n_imgs]]
        if not images_cpu or images_cpu[0] is None:
            raise FileNotFoundError(f"Could not read images from path: {path} with list: {images_list[0] if images_list else 'empty'}")
        return images_cpu 

    def convert_images(self, left_imgs_cpu, right_imgs_cpu, apply_clahe=False, tile=11, climp=5.0, undist=False):
        processed_left_imgs = []
        processed_right_imgs = []

        clahe = None
        if apply_clahe:
            clahe = cv2.createCLAHE(clipLimit=climp, tileGridSize=(tile, tile))

        for img_l, img_r in zip(left_imgs_cpu, right_imgs_cpu):
            img_l_processed = img_l
            img_r_processed = img_r
            if apply_clahe:
                img_l_processed = clahe.apply(img_l_processed)
                img_r_processed = clahe.apply(img_r_processed)
            
            if undist:
                k_l_cpu = cp.asnumpy(self.camera_params['left']['kk'])
                kc_l_cpu = cp.asnumpy(self.camera_params['left']['kc'])
                k_r_cpu = cp.asnumpy(self.camera_params['right']['kk'])
                kc_r_cpu = cp.asnumpy(self.camera_params['right']['kc'])
                
                img_l_processed = cv2.undistort(img_l_processed, k_l_cpu, kc_l_cpu)
                img_r_processed = cv2.undistort(img_r_processed, k_r_cpu, kc_r_cpu)
            
            processed_left_imgs.append(img_l_processed)
            processed_right_imgs.append(img_r_processed)
        
        if not processed_left_imgs or not processed_right_imgs:
             raise ValueError("Image processing resulted in empty image lists.")

        stacked_left_np = np.stack(processed_left_imgs, axis=-1).astype(np.uint8)
        stacked_right_np = np.stack(processed_right_imgs, axis=-1).astype(np.uint8)

        self.left_images = cp.asarray(stacked_left_np)
        self.right_images = cp.asarray(stacked_right_np)
        
        del processed_left_imgs, processed_right_imgs, stacked_left_np, stacked_right_np
        gc.collect()
        return True


    def set_datalimit(self):
        device_id = 0
        cp.cuda.Device(device_id).use()
        total_memory = cp.cuda.runtime.getDeviceProperties(device_id)['totalGlobalMem']
        return total_memory / (1024 ** 3) 

    def estimate_batch_size(self, Kx, Ky, Nz, T, safety_margin=0.8): 
        mem_target_GB = self.max_gpu_usage_gb * safety_margin 
        
        bytes_per_float32 = 4
        mem_target_bytes = mem_target_GB * (1024 ** 3)

        mem_per_kernel_for_correlation_inputs = Kx * Ky * Nz * T * 2 * bytes_per_float32
        
        if mem_per_kernel_for_correlation_inputs == 0: 
            return 1000 

        batch_size_Nc = int(mem_target_bytes // mem_per_kernel_for_correlation_inputs)

        return max(1, batch_size_Nc)

    def transform_gcs2ccs(self, points_3d_gpu, cam_name): 
        k = self.camera_params[cam_name]['kk']      
        rot = self.camera_params[cam_name]['r']    
        tran = self.camera_params[cam_name]['t']   

        num_points = points_3d_gpu.shape[0]
        if num_points == 0:
            return cp.empty((2, 0), dtype=cp.float32)

        ones = cp.ones((num_points, 1), dtype=cp.float32)
        xyz_gcs_1 = cp.hstack((points_3d_gpu, ones))

        rt_matrix = cp.hstack((rot, tran)) 

        xyz_ccs_homogeneous = cp.dot(rt_matrix, xyz_gcs_1.T).T
        del xyz_gcs_1, rt_matrix, ones
        
        zc = xyz_ccs_homogeneous[:, 2]
        epsilon = 1e-10
        valid_zc_mask = zc > epsilon 
        
        uv_points_cam = cp.empty((num_points, 2), dtype=cp.float32) 
        
        if cp.any(valid_zc_mask):
            xn = xyz_ccs_homogeneous[valid_zc_mask, 0] / zc[valid_zc_mask]
            yn = xyz_ccs_homogeneous[valid_zc_mask, 1] / zc[valid_zc_mask]

            u_valid = k[0,0] * xn + k[0,2]
            v_valid = k[1,1] * yn + k[1,2]
            
            uv_points_cam[valid_zc_mask, 0] = u_valid
            uv_points_cam[valid_zc_mask, 1] = v_valid

        uv_points_cam[~valid_zc_mask, :] = -1 
        
        del xyz_ccs_homogeneous, zc, valid_zc_mask
        return uv_points_cam.T 

    def points3d(self, x_lim, y_lim, z_lim, xy_step, z_step):
        x_lin_np = np.arange(x_lim[0], x_lim[1] + xy_step, xy_step, dtype=np.float16)
        y_lin_np = np.arange(y_lim[0], y_lim[1] + xy_step, xy_step, dtype=np.float16)
        z_lin_np = np.arange(z_lim[0], z_lim[1] + z_step, z_step, dtype=np.float16)

        self.x_vals = cp.asarray(x_lin_np)
        self.y_vals = cp.asarray(y_lin_np)
        self.z_vals = cp.asarray(z_lin_np)

        X, Y, Z = cp.meshgrid(self.x_vals, self.y_vals, self.z_vals, indexing='ij')
        self.grid = cp.stack((X, Y, Z), axis=-1)  
        del X, Y, Z
        gc.collect()


    def bi_interpolation(self, images_gpu, uv_points_gpu, batch_size_interp=2000000): 
        if uv_points_gpu.shape[1] == 0:
            return cp.empty((0, images_gpu.shape[2]), dtype=cp.float32), \
                   cp.empty((0, images_gpu.shape[2]), dtype=cp.float32)

        images_gpu = cp.asarray(images_gpu) 
        uv_points_gpu = cp.asarray(uv_points_gpu)

        if len(images_gpu.shape) == 2: 
            images_gpu = images_gpu[:, :, cp.newaxis]

        height, width, num_images_T = images_gpu.shape
        N_total_points = uv_points_gpu.shape[1]

        interpolated_all = cp.empty((N_total_points, num_images_T), dtype=cp.float32)
        std_all = cp.empty((N_total_points, num_images_T), dtype=cp.float32)

        for i in range(0, N_total_points, batch_size_interp):
            end = min(i + batch_size_interp, N_total_points)
            uv_batch = uv_points_gpu[:, i:end]

            x = uv_batch[0].astype(cp.float32) 
            y = uv_batch[1].astype(cp.float32) 

            x = cp.clip(x, 0, width - 1 - epsilon)
            y = cp.clip(y, 0, height - 1 - epsilon)

            x1 = cp.floor(x).astype(cp.int32)
            y1 = cp.floor(y).astype(cp.int32)
            x2 = cp.clip(x1 + 1, 0, width - 1)
            y2 = cp.clip(y1 + 1, 0, height - 1)
            
            wa = (x - x1) 
            wb = (y - y1)
            
            current_batch_N_pts = x.shape[0]
            batch_interpolated_T = cp.empty((current_batch_N_pts, num_images_T), dtype=cp.float32)
            batch_std_T = cp.empty((current_batch_N_pts, num_images_T), dtype=cp.float32)

            for k_img_idx in range(num_images_T):
                img_slice_k = images_gpu[:, :, k_img_idx] 
                
                p11 = img_slice_k[y1, x1]
                p12 = img_slice_k[y2, x1]
                p21 = img_slice_k[y1, x2]
                p22 = img_slice_k[y2, x2]

                interp_val = p11 * (1 - wa) * (1 - wb) + \
                             p21 * wa * (1 - wb) + \
                             p12 * (1 - wa) * wb + \
                             p22 * wa * wb
                
                batch_interpolated_T[:, k_img_idx] = interp_val

                points_for_std = cp.stack((p11, p12, p21, p22), axis=0) 
                batch_std_T[:, k_img_idx] = cp.std(points_for_std, axis=0)
                del p11, p12, p21, p22, points_for_std, interp_val

            interpolated_all[i:end, :] = batch_interpolated_T
            std_all[i:end, :] = batch_std_T
            
            del x, y, x1, y1, x2, y2, wa, wb, uv_batch, batch_interpolated_T, batch_std_T

        return interpolated_all, std_all


    def filter_sparse_points(self, xyz, corr, min_neighbors=5, radius=10):


        if xyz.shape[0] == 0:
            return cp.array([]), cp.array([])
        
        db = DBSCAN(eps=radius, min_samples=min_neighbors).fit(xyz)

        mask = db.labels_ != -1

        xyz_filtered = xyz[mask]
        corr_filtered = corr[mask]

        return cp.asarray(xyz_filtered), cp.asarray(corr_filtered)

    def spatial_temp_correl(self, interp_L_kernels, interp_R_kernels):
        Current_Nc_Batch, Kx, Ky, Nz, T = interp_L_kernels.shape
        
        if Current_Nc_Batch == 0:
             return cp.empty((0, Nz), dtype=cp.float32), \
                    cp.empty((0,), dtype=cp.float32), \
                    cp.empty((0,), dtype=cp.float32) 
        
        K_features = Kx * Ky * T
        if K_features == 0: 
            corr_values = cp.zeros((Current_Nc_Batch, Nz), dtype=cp.float32)
        else:
            L_flat = interp_L_kernels.transpose(0, 3, 1, 2, 4).reshape(Current_Nc_Batch * Nz, K_features)
            R_flat = interp_R_kernels.transpose(0, 3, 1, 2, 4).reshape(Current_Nc_Batch * Nz, K_features)

            L_mu = cp.mean(L_flat, axis=1, keepdims=True)
            R_mu = cp.mean(R_flat, axis=1, keepdims=True)

            Lz = L_flat - L_mu
            Rz = R_flat - R_mu
            del L_mu, R_mu, L_flat, R_flat

            numerator = cp.sum(Lz * Rz, axis=1)
            denominator_L_sq_sum = cp.sum(Lz ** 2, axis=1)
            denominator_R_sq_sum = cp.sum(Rz ** 2, axis=1)
            del Lz, Rz
            
            denominator = cp.sqrt(denominator_L_sq_sum * denominator_R_sq_sum)
            del denominator_L_sq_sum, denominator_R_sq_sum

            corr_flat = numerator / cp.maximum(denominator, 1e-10)
            corr_values = corr_flat.reshape(Current_Nc_Batch, Nz)
            del corr_flat, numerator, denominator
        
        corr_max_for_each_kernel = cp.nanmax(corr_values, axis=1)

        z_best_indices = cp.nanargmax(corr_values, axis=1)
        
        z_best_actual_values = self.z_vals[z_best_indices]

        return corr_values, corr_max_for_each_kernel, z_best_actual_values

    def process(self, Kx=5, Ky=5, stride=1, nc_batch_size=None, y_coord_offset_val=0.0):

        Nx, Ny_current_grid, Nz_current_grid = self.grid.shape[:3]
        T = self.left_images.shape[2]

        if nc_batch_size is None:
            nc_batch_size = self.estimate_batch_size(Kx, Ky, Nz_current_grid, T, safety_margin=0.35)
            print(f"[INFO] Using estimated Nc_batch_size = {nc_batch_size}")

        pad_x = Kx // 2
        pad_y = Ky // 2

        ix_centers_all = cp.arange(pad_x, Nx - pad_x, stride, dtype=cp.int32)
        iy_centers_all = cp.arange(pad_y, Ny_current_grid - pad_y, stride, dtype=cp.int32)
        
        if len(ix_centers_all) == 0 or len(iy_centers_all) == 0 :
            print("[WARN] No valid kernel centers found. Check Kx, Ky, stride, and grid dimensions.")
            return cp.empty((0,3), dtype=cp.float32), cp.empty((0,), dtype=cp.float32), \
                   cp.empty((0,Nz_current_grid), dtype=cp.float32), \
                   cp.empty((0,), dtype=cp.float32), cp.empty((0,), dtype=cp.float32)

        IX_all, IY_all = cp.meshgrid(ix_centers_all, iy_centers_all, indexing='ij')
        IX_all = IX_all.ravel() 
        IY_all = IY_all.ravel() 
        Nc_total = IX_all.shape[0]

        if Nc_total == 0:
            print("[WARN] Nc_total is 0. No kernels to process.")
            return cp.empty((0,3), dtype=cp.float32), cp.empty((0,), dtype=cp.float32), \
                   cp.empty((0,Nz_current_grid), dtype=cp.float32), \
                   cp.empty((0,), dtype=cp.float32), cp.empty((0,), dtype=cp.float32)


        xyz_final_parts_cpu = []
        corr_max_parts_cpu = []
        corr_all_volume_parts_cpu = [] 
        stdL_final_parts_cpu = []
        stdR_final_parts_cpu = []

        off_x_kernel = cp.arange(-pad_x, pad_x + 1, dtype=cp.int32) 
        off_y_kernel = cp.arange(-pad_y, pad_y + 1, dtype=cp.int32) 

        print(f"[INFO] Total kernels to process (Nc_total): {Nc_total}. Batch size (nc_batch_size): {nc_batch_size}")

        for i in range(0, Nc_total, 30):
            batch_start_time = cp.cuda.Event()
            batch_end_time = cp.cuda.Event()
            batch_start_time.record()

            current_batch_idx_start = i
            current_batch_idx_end = min(i + nc_batch_size, Nc_total)
            current_Nc_in_batch = current_batch_idx_end - current_batch_idx_start

            IX_batch_centers = IX_all[current_batch_idx_start:current_batch_idx_end] 
            IY_batch_centers = IY_all[current_batch_idx_start:current_batch_idx_end]
            
            kernel_voxels_list = []
            for nc_idx in range(current_Nc_in_batch):
                center_ix = IX_batch_centers[nc_idx]
                center_iy = IY_batch_centers[nc_idx]
                
                abs_x_indices_kernel = center_ix + off_x_kernel 
                abs_y_indices_kernel = center_iy + off_y_kernel 

                kernel_slice = self.grid[abs_x_indices_kernel[0]:abs_x_indices_kernel[-1]+1,
                                         abs_y_indices_kernel[0]:abs_y_indices_kernel[-1]+1,
                                         :, :]
                kernel_voxels_list.append(kernel_slice)

            kernel_voxel_coords = cp.stack(kernel_voxels_list, axis=0).astype(cp.float32)
            del kernel_voxels_list

            num_total_voxels_in_batch_kernels = current_Nc_in_batch * Kx * Ky * Nz_current_grid
            kernel_voxel_coords_flat = kernel_voxel_coords.reshape(num_total_voxels_in_batch_kernels, 3)
            # del kernel_voxel_coords

            uv_left_batch_flat = self.transform_gcs2ccs(kernel_voxel_coords_flat, cam_name='left')
            uv_right_batch_flat = self.transform_gcs2ccs(kernel_voxel_coords_flat, cam_name='right')
            # del kernel_voxel_coords_flat

            interp_L_flat, std_L_val_flat = self.bi_interpolation(self.left_images, uv_left_batch_flat)
            interp_R_flat, std_R_val_flat = self.bi_interpolation(self.right_images, uv_right_batch_flat)
            #del uv_left_batch_flat, uv_right_batch_flat

            interp_L_k = interp_L_flat.reshape(current_Nc_in_batch, Kx, Ky, Nz_current_grid, T)
            interp_R_k = interp_R_flat.reshape(current_Nc_in_batch, Kx, Ky, Nz_current_grid, T)
            del interp_L_flat, interp_R_flat
            
            std_L_k_texture = std_L_val_flat.reshape(current_Nc_in_batch, Kx, Ky, Nz_current_grid, T)
            std_R_k_texture = std_R_val_flat.reshape(current_Nc_in_batch, Kx, Ky, Nz_current_grid, T)
            del std_L_val_flat, std_R_val_flat

            corr_all_plane_batch, corr_max_kernel_batch, z_best_batch_values = self.spatial_temp_correl(
                interp_L_k, interp_R_k
            ) 
            del interp_L_k, interp_R_k

            z_best_indices_in_Nz = cp.nanargmax(corr_all_plane_batch, axis=1) 
            selected_std_L_list = []
            selected_std_R_list = []
            for ker_idx_in_batch in range(current_Nc_in_batch):
                z_idx = z_best_indices_in_Nz[ker_idx_in_batch]
                selected_std_L_list.append(std_L_k_texture[ker_idx_in_batch, :, :, z_idx, :]) 
                selected_std_R_list.append(std_R_k_texture[ker_idx_in_batch, :, :, z_idx, :])
            
            if selected_std_L_list:
                std_L_at_z_best_KxKyT = cp.stack(selected_std_L_list, axis=0) 
                std_R_at_z_best_KxKyT = cp.stack(selected_std_R_list, axis=0)
                
                mean_std_L_batch = cp.mean(std_L_at_z_best_KxKyT, axis=(1,2,3)) 
                mean_std_R_batch = cp.mean(std_R_at_z_best_KxKyT, axis=(1,2,3))
                del std_L_at_z_best_KxKyT, std_R_at_z_best_KxKyT
            else: 
                mean_std_L_batch = cp.empty((0,),dtype=cp.float32)
                mean_std_R_batch = cp.empty((0,),dtype=cp.float32)

            del std_L_k_texture, std_R_k_texture, selected_std_L_list, selected_std_R_list

            x_coords_batch = self.x_vals[IX_batch_centers] 
            y_coords_batch = self.y_vals[IY_batch_centers] + y_coord_offset_val 
            
            xyz_batch_gpu = cp.stack([x_coords_batch, y_coords_batch, z_best_batch_values], axis=1)

            xyz_final_parts_cpu.append(cp.asnumpy(xyz_batch_gpu))
            corr_max_parts_cpu.append(cp.asnumpy(corr_max_kernel_batch))
            corr_all_volume_parts_cpu.append(cp.asnumpy(corr_all_plane_batch))
            stdL_final_parts_cpu.append(cp.asnumpy(mean_std_L_batch))
            stdR_final_parts_cpu.append(cp.asnumpy(mean_std_R_batch))

            del xyz_batch_gpu, corr_max_kernel_batch, corr_all_plane_batch, z_best_batch_values
            del mean_std_L_batch, mean_std_R_batch, z_best_indices_in_Nz
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()
            
            batch_end_time.record()
            batch_end_time.synchronize()
            processing_time_ms = cp.cuda.get_elapsed_time(batch_start_time, batch_end_time)
            print(f"  Batch {i//nc_batch_size + 1}/{Nc_total//nc_batch_size + 1 if Nc_total > 0 and nc_batch_size > 0 else 1} "
                  f"({current_Nc_in_batch} kernels) processed in {processing_time_ms:.2f} ms. "
                  f"Mem Free: {cp.get_default_memory_pool().free_bytes() / (1024**2):.2f} MB / "
                  f"Total: {cp.get_default_memory_pool().total_bytes() / (1024**2):.2f} MB")


        if not xyz_final_parts_cpu: 
             print("[WARN] No data parts to concatenate. Returning empty arrays.")
             return cp.empty((0,3), dtype=cp.float32), cp.empty((0,), dtype=cp.float32), \
                   cp.empty((0,Nz_current_grid), dtype=cp.float32), \
                   cp.empty((0,), dtype=cp.float32), cp.empty((0,), dtype=cp.float32)

        xyz_final_all_cpu = np.concatenate(xyz_final_parts_cpu, axis=0)
        corr_max_all_cpu = np.concatenate(corr_max_parts_cpu, axis=0)
        corr_all_volume_final_cpu = np.concatenate(corr_all_volume_parts_cpu, axis=0)
        stdL_final_all_cpu = np.concatenate(stdL_final_parts_cpu, axis=0)
        stdR_final_all_cpu = np.concatenate(stdR_final_parts_cpu, axis=0)

        return cp.asarray(xyz_final_all_cpu), cp.asarray(corr_max_all_cpu), \
               cp.asarray(corr_all_volume_final_cpu), \
               cp.asarray(stdL_final_all_cpu), cp.asarray(stdR_final_all_cpu)

    def process_segmented_y(self, Kx=5, Ky=5, stride=1, nc_batch_size_param=None, block_size_y_voxels=10): 
        Ny_total = self.grid.shape[1]
        if Ny_total == 0: return 
        
        xyz_all_parts = []
        corr_max_all_parts = []
        corrmap_all_parts = [] 
        stdL_all_parts = []
        stdR_all_parts = []
        
        grid_backup = self.grid
        y_vals_backup = self.y_vals

        for y0_idx in range(0, Ny_total, block_size_y_voxels):
            y1_idx = min(y0_idx + block_size_y_voxels, Ny_total)
            if y0_idx == y1_idx: continue 

            print(f"[Y-SEGMENT] Processing Y-slice: indices {y0_idx} to {y1_idx-1}")

            self.grid = grid_backup[:, y0_idx:y1_idx, :, :]
            self.y_vals = y_vals_backup[y0_idx:y1_idx]

            y_offset_mm_for_slice = y_vals_backup[y0_idx] if y0_idx > 0 else 0.0

            xyz, corr_max, corrmap_slice, stdL, stdR = self.process(
                Kx=Kx, Ky=Ky, stride=stride, nc_batch_size=nc_batch_size_param,
                y_coord_offset_val=y_vals_backup[y0_idx] 
            )

            if xyz.shape[0] > 0: 
                xyz_all_parts.append(xyz)
                corr_max_all_parts.append(corr_max)
                corrmap_all_parts.append(corrmap_slice) 
                stdL_all_parts.append(stdL)
                stdR_all_parts.append(stdR)

            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()

        self.grid = grid_backup 
        self.y_vals = y_vals_backup

        if not xyz_all_parts:
             return cp.empty((0,3)), cp.empty((0,)), cp.empty((0,self.grid.shape[2])), cp.empty((0,)), cp.empty((0,))

        xyz_final = cp.concatenate(xyz_all_parts, axis=0)
        corr_final = cp.concatenate(corr_max_all_parts, axis=0)
        if corrmap_all_parts and all(c.shape[1] == corrmap_all_parts[0].shape[1] for c in corrmap_all_parts if c.ndim == 2 and c.size > 0): # Check Nz dim
            corrmap_final = cp.concatenate(corrmap_all_parts, axis=0)
        else: 
            Nz_total = self.grid.shape[2] if self.grid.ndim >=3 else 0
            num_total_points = xyz_final.shape[0]
            corrmap_final = cp.full((num_total_points, Nz_total), cp.nan, dtype=cp.float32) if num_total_points > 0 else cp.empty((0,Nz_total), dtype=cp.float32)

        stdL_final = cp.concatenate(stdL_all_parts, axis=0)
        stdR_final = cp.concatenate(stdR_all_parts, axis=0)

        return xyz_final, corr_final, corrmap_final, stdL_final, stdR_final


    def process_segmented_z(self, Kx=5, Ky=5, stride=1, nc_batch_size_param=None, Nz_block_voxels=50):
        Nx, Ny, Nz_total = self.grid.shape[:3]
        if Nz_total == 0: return 
            
        pad_x = Kx // 2
        pad_y = Ky // 2
        ix_centers_all_xy = cp.arange(pad_x, Nx - pad_x, stride, dtype=cp.int32)
        iy_centers_all_xy = cp.arange(pad_y, Ny - pad_y, stride, dtype=cp.int32)
        
        if len(ix_centers_all_xy) == 0 or len(iy_centers_all_xy) == 0:
            print("[WARN] No valid XY kernel centers for Z-segmentation.")
            return cp.empty((0,3)), cp.empty((0,)), cp.empty((0,Nz_total)), cp.empty((0,)), cp.empty((0,))

        IX_global_centers, IY_global_centers = cp.meshgrid(ix_centers_all_xy, iy_centers_all_xy, indexing='ij')
        IX_global_centers = IX_global_centers.ravel()
        IY_global_centers = IY_global_centers.ravel()
        Nc_for_xy_plane = IX_global_centers.shape[0]

        if Nc_for_xy_plane == 0:
            print("[WARN] Nc_for_xy_plane is 0 in Z-segmentation.")
            return cp.empty((0,3)), cp.empty((0,)), cp.empty((0,Nz_total)), cp.empty((0,)), cp.empty((0,))

        corr_map_overall_z = cp.full((Nc_for_xy_plane, Nz_total), -cp.inf, dtype=cp.float32)
        stdL_overall_best_z = cp.full((Nc_for_xy_plane,), cp.nan, dtype=cp.float32)
        stdR_overall_best_z = cp.full((Nc_for_xy_plane,), cp.nan, dtype=cp.float32)
        
        grid_backup = self.grid
        z_vals_backup = self.z_vals

        for z0_idx in range(0, Nz_total, Nz_block_voxels):
            z1_idx = min(z0_idx + Nz_block_voxels, Nz_total)
            if z0_idx == z1_idx: continue

            print(f"[Z-SEGMENT] Processing Z-slice: indices {z0_idx} to {z1_idx-1}")
            
            self.grid = grid_backup[:, :, z0_idx:z1_idx, :]
            self.z_vals = z_vals_backup[z0_idx:z1_idx] 
            current_Nz_in_slice = self.grid.shape[2]

            xyz_slice, corr_max_slice, corrmap_slice, stdL_slice, stdR_slice = self.process(
                Kx=Kx, Ky=Ky, stride=stride, nc_batch_size=nc_batch_size_param,
                y_coord_offset_val=0.0 
            )

            if xyz_slice.shape[0] == Nc_for_xy_plane: 
                corr_map_overall_z[:, z0_idx:z1_idx] = corrmap_slice
        
            else:
                print(f"[WARN] Z-Slice {z0_idx}-{z1_idx}: Mismatch in Nc. Expected {Nc_for_xy_plane}, got {xyz_slice.shape[0]}. Skipping update for this slice.")

            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()

        self.grid = grid_backup
        self.z_vals = z_vals_backup

        corr_max_overall = cp.nanmax(corr_map_overall_z, axis=1) 
        z_best_indices_overall = cp.nanargmax(corr_map_overall_z, axis=1) 
        z_best_values_overall = self.z_vals[z_best_indices_overall] 

        x_coords_final = self.x_vals[IX_global_centers]
        y_coords_final = self.y_vals[IY_global_centers]
        
        xyz_final = cp.stack([x_coords_final, y_coords_final, z_best_values_overall], axis=1)

        stdL_final = cp.full_like(corr_max_overall, cp.nan) 
        stdR_final = cp.full_like(corr_max_overall, cp.nan)

        return xyz_final, corr_max_overall, corr_map_overall_z, stdL_final, stdR_final

epsilon = 1e-6