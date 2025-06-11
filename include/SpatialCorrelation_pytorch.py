import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import os
import matplotlib.pyplot as plt
import cv2
from scipy.spatial import cKDTree

class PyTorchStereoCorrel(nn.Module):
    def __init__(self, yaml_file):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"PyTorch a ser executado no dispositivo: {self.device}")

        self.left_images = torch.empty(0)
        self.right_images = torch.empty(0)

        self.camera_params = {
            'left': {'kk': None, 'kc': None, 'r': None, 't': None},
            'right': {'kk': None, 'kc': None, 'r': None, 't': None},
            'stereo': {'R': None, 'T': None}
        }
        self.read_yaml_file(yaml_file)

        self.x_vals = torch.empty(0)
        self.y_vals = torch.empty(0)
        self.z_vals = torch.empty(0)
        self.grid = torch.empty(0)
        self.epsilon = 1e-10

    def read_yaml_file(self, yaml_file):
        with open(yaml_file) as file:
            params = yaml.safe_load(file)

        for cam in ['left', 'right']:
            self.camera_params[cam]['kk'] = torch.tensor(params[f'camera_matrix_{cam}'], dtype=torch.float32, device=self.device)
            self.camera_params[cam]['kc'] = torch.tensor(params[f'dist_coeffs_{cam}'], dtype=torch.float32, device=self.device)
            self.camera_params[cam]['r'] = torch.tensor(params[f'rot_matrix_{cam}'], dtype=torch.float32, device=self.device)
            self.camera_params[cam]['t'] = torch.tensor(params[f't_{cam}'], dtype=torch.float32, device=self.device).view(3, 1)
        
        self.camera_params['stereo']['R'] = torch.tensor(params['R'], dtype=torch.float32, device=self.device)
        self.camera_params['stereo']['T'] = torch.tensor(params['T'], dtype=torch.float32, device=self.device).view(3, 1)

    def convert_images(self, left_imgs_cpu, right_imgs_cpu, apply_clahe=True, undist=True):
        processed_left_imgs = []
        processed_right_imgs = []
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(11, 11))

        for img_l, img_r in zip(left_imgs_cpu, right_imgs_cpu):
            if apply_clahe:
                img_l = clahe.apply(img_l)
                img_r = clahe.apply(img_r)
            if undist:
                k_l_cpu = self.camera_params['left']['kk'].cpu().numpy()
                kc_l_cpu = self.camera_params['left']['kc'].cpu().numpy()
                k_r_cpu = self.camera_params['right']['kk'].cpu().numpy()
                kc_r_cpu = self.camera_params['right']['kc'].cpu().numpy()
                img_l = cv2.undistort(img_l, k_l_cpu, kc_l_cpu)
                img_r = cv2.undistort(img_r, k_r_cpu, kc_r_cpu)
            processed_left_imgs.append(img_l)
            processed_right_imgs.append(img_r)

        self.left_images = torch.from_numpy(np.stack(processed_left_imgs, axis=0)).to(self.device, dtype=torch.float32)
        self.right_images = torch.from_numpy(np.stack(processed_right_imgs, axis=0)).to(self.device, dtype=torch.float32)

    def points3d(self, x_lim, y_lim, z_lim, xy_step, z_step):
        self.x_vals = torch.arange(x_lim[0], x_lim[1] + xy_step, xy_step, dtype=torch.float16, device=self.device)
        self.y_vals = torch.arange(y_lim[0], y_lim[1] + xy_step, xy_step, dtype=torch.float16, device=self.device)
        self.z_vals = torch.arange(z_lim[0], z_lim[1] + z_step, z_step, dtype=torch.float16, device=self.device)
        
        X, Y, Z = torch.meshgrid(self.x_vals, self.y_vals, self.z_vals, indexing='ij')
        self.grid = torch.stack((X, Y, Z), axis=-1)

    def transform_gcs2ccs(self, points_3d, cam_name):
        k, r, t = self.camera_params[cam_name]['kk'], self.camera_params[cam_name]['r'], self.camera_params[cam_name]['t']
        
        num_points = points_3d.shape[0]
        if num_points == 0:
            return torch.empty((0, 2), device=self.device)

        ones = torch.ones((num_points, 1), device=self.device, dtype=points_3d.dtype)
        xyz_gcs_1 = torch.cat([points_3d, ones], dim=1)
        rt_matrix = torch.cat([r, t], dim=1)
        xyz_ccs = torch.matmul(rt_matrix, xyz_gcs_1.T).T
        
        zc = xyz_ccs[:, 2]
        valid_mask = zc > self.epsilon
        uv_points = torch.full((num_points, 2), -1.0, device=self.device, dtype=torch.float32)
        
        if torch.any(valid_mask):
            xn = xyz_ccs[valid_mask, 0] / zc[valid_mask]
            yn = xyz_ccs[valid_mask, 1] / zc[valid_mask]
            
            u = k[0, 0] * xn + k[0, 2]
            v = k[1, 1] * yn + k[1, 2]
            
            uv_points[valid_mask] = torch.stack([u, v], dim=1)
        return uv_points

    def interpolate_images(self, images, uv_points):
        if uv_points.numel() == 0:
            return torch.empty((0, images.shape[0]), device=self.device)
        
        T, H, W = images.shape
        N = uv_points.shape[0]

        u_norm = (uv_points[:, 0] / (W - 1)) * 2 - 1
        v_norm = (uv_points[:, 1] / (H - 1)) * 2 - 1
        
        grid = torch.stack([u_norm, v_norm], dim=1).view(1, N, 1, 2)
        images_batch = images.unsqueeze(0)

        interpolated = F.grid_sample(images_batch, grid, mode='bilinear', padding_mode='border', align_corners=True)
        return interpolated.view(T, N).T

    def zncc_correlation(self, L_patches, R_patches):
        L_mean = torch.mean(L_patches, dim=1, keepdim=True)
        R_mean = torch.mean(R_patches, dim=1, keepdim=True)
        L_centered = L_patches - L_mean
        R_centered = R_patches - R_mean

        numerator = torch.sum(L_centered * R_centered, dim=1)
        denom_L = torch.sum(L_centered**2, dim=1)
        denom_R = torch.sum(R_centered**2, dim=1)
        denominator = torch.sqrt(denom_L * denom_R)
        
        return numerator / (denominator + self.epsilon)

    def process_segmented_z(self, Kx, Ky, stride=1, Nz_block_voxels=40):
        Nx, Ny, Nz_total = self.grid.shape[:3]
        T = self.left_images.shape[0]
        
        pad_x, pad_y = Kx // 2, Ky // 2
        ix_centers = torch.arange(pad_x, Nx - pad_x, stride, device=self.device)
        iy_centers = torch.arange(pad_y, Ny - pad_y, stride, device=self.device)

        if len(ix_centers) == 0 or len(iy_centers) == 0:
            return torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)

        IX_centers, IY_centers = torch.meshgrid(ix_centers, iy_centers, indexing='ij')
        IX_centers, IY_centers = IX_centers.ravel(), IY_centers.ravel()
        Nc_for_xy_plane = IX_centers.shape[0]

        corr_map_overall_z = torch.full((Nc_for_xy_plane, Nz_total), -torch.inf, device=self.device, dtype=torch.float32)

        for z0_idx in range(0, Nz_total, Nz_block_voxels):
            z1_idx = min(z0_idx + Nz_block_voxels, Nz_total)
            print(f"[Z-SEGMENT] Processando Z-slice: índices {z0_idx} a {z1_idx-1}")
            
            grid_slice = self.grid[:, :, z0_idx:z1_idx, :].to(torch.float32)
            current_Nz_in_slice = grid_slice.shape[2]

            grid_flat_xy = grid_slice.permute(2,0,1,3).reshape(current_Nz_in_slice, Nx*Ny, 3)
            
            uv_left = self.transform_gcs2ccs(grid_flat_xy.reshape(-1, 3), 'left')
            uv_right = self.transform_gcs2ccs(grid_flat_xy.reshape(-1, 3), 'right')

            interp_L = self.interpolate_images(self.left_images, uv_left)
            interp_R = self.interpolate_images(self.right_images, uv_right)
            
            interp_L = interp_L.view(current_Nz_in_slice, Nx, Ny, T).permute(3,0,1,2)
            interp_R = interp_R.view(current_Nz_in_slice, Nx, Ny, T).permute(3,0,1,2)

            L_unfold = F.unfold(interp_L.permute(1,0,2,3).reshape(current_Nz_in_slice, T, Nx, Ny), kernel_size=(Kx, Ky), stride=(stride, stride))
            R_unfold = F.unfold(interp_R.permute(1,0,2,3).reshape(current_Nz_in_slice, T, Nx, Ny), kernel_size=(Kx, Ky), stride=(stride, stride))
            
            L_patches = L_unfold.permute(2, 1, 0).reshape(Nc_for_xy_plane, -1, current_Nz_in_slice)
            R_patches = R_unfold.permute(2, 1, 0).reshape(Nc_for_xy_plane, -1, current_Nz_in_slice)
            
            for z_local_idx in range(current_Nz_in_slice):
                corr_slice = self.zncc_correlation(L_patches[:,:,z_local_idx], R_patches[:,:,z_local_idx])
                corr_map_overall_z[:, z0_idx + z_local_idx] = corr_slice

        corr_max_overall, z_best_indices_overall = torch.max(corr_map_overall_z, dim=1)
        z_best_values_overall = self.z_vals[z_best_indices_overall]

        x_coords_final = self.x_vals[IX_centers]
        y_coords_final = self.y_vals[IY_centers]
        
        xyz_final = torch.stack([x_coords_final, y_coords_final, z_best_values_overall], dim=1).to(torch.float32)

        return xyz_final, corr_max_overall, corr_map_overall_z, None, None

    def filter_sparse_points(self, xyz_gpu, corr_gpu, min_neighbors=5, radius=10):
        if xyz_gpu.numel() == 0:
            return xyz_gpu, corr_gpu
        
        xyz_cpu = xyz_gpu.cpu().numpy()
        corr_cpu = corr_gpu.cpu().numpy()

        tree = cKDTree(xyz_cpu)
        neighbor_counts = np.array([len(neighbors) for neighbors in tree.query_ball_point(xyz_cpu, r=radius)])
        dense_mask = neighbor_counts >= min_neighbors

        return torch.from_numpy(xyz_cpu[dense_mask]).to(self.device), torch.from_numpy(corr_cpu[dense_mask]).to(self.device)

    def plot_3d_points(self, x, y, z, color=None, title='Plot 3D'):
        def to_numpy(tensor):
            if isinstance(tensor, torch.Tensor):
                return tensor.cpu().numpy()
            return tensor
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.title.set_text(title)

        scatter = ax.scatter(to_numpy(x), to_numpy(y), to_numpy(z), c=to_numpy(color), cmap='viridis', marker='o')
        plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
        ax.set_xlabel('X [mm]'); ax.set_ylabel('Y [mm]'); ax.set_zlabel('Z [mm]')
        ax.set_aspect('equal', adjustable='box')
        plt.show()