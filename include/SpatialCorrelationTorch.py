# Versão convertida de CuPy para PyTorch
import torch
import torch.nn.functional as F
import numpy as np
import yaml
import os
import matplotlib.pyplot as plt
import cv2
import gc
from scipy.spatial import cKDTree

class StereoTemporalSpatialCorrel:
    def __init__(self, yaml_file):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.left_images = torch.tensor([], device=self.device)
        self.right_images = torch.tensor([], device=self.device)

        self.camera_params = {
            'left': {'kk': np.array([]), 'kc': np.array([]), 'r': np.array([]), 't': np.array([])},
            'right': {'kk': np.array([]), 'kc': np.array([]), 'r': np.array([]), 't': np.array([])},
            'stereo': {'R': np.array([]), 'T': np.array([])}
        }
        self.read_yaml_file(yaml_file)
        self.max_gpu_usage = 8 // 5

    def read_yaml_file(self, yaml_file):
        with open(yaml_file) as file:
            params = yaml.safe_load(file)

        for cam in ['left', 'right']:
            self.camera_params[cam]['kk'] = np.array(params[f'camera_matrix_{cam}'], dtype=np.float32)
            self.camera_params[cam]['kc'] = np.array(params[f'dist_coeffs_{cam}'], dtype=np.float32)
            self.camera_params[cam]['r'] = np.array(params[f'rot_matrix_{cam}'], dtype=np.float32)
            self.camera_params[cam]['t'] = np.array(params[f't_{cam}'], dtype=np.float32)

        self.camera_params['stereo']['R'] = np.array(params['R'], dtype=np.float32)
        self.camera_params['stereo']['T'] = np.array(params['T'], dtype=np.float32)

    def read_images(self, path, images_list, n_imgs):
        images = [cv2.imread(os.path.join(path, str(img_name)), cv2.IMREAD_GRAYSCALE)
                  for img_name in images_list[0:n_imgs]]
        return images

    def remove_img_distortion(self, img, camera):
        return cv2.undistort(img, self.camera_params[camera]['kk'], self.camera_params[camera]['kc'])

    def convert_images(self, left_imgs, right_imgs, apply_clahe=False, tile=11, climp=5.0, undist=False):
        if apply_clahe:
            clahe = cv2.createCLAHE(clipLimit=climp, tileGridSize=(tile, tile))
            if undist:
                left_imgs = [self.remove_img_distortion(clahe.apply(img), 'left') for img in left_imgs]
                right_imgs = [self.remove_img_distortion(clahe.apply(img), 'right') for img in right_imgs]
            else:
                left_imgs = [clahe.apply(img) for img in left_imgs]
                right_imgs = [clahe.apply(img) for img in right_imgs]

        self.left_images = torch.from_numpy(np.stack(left_imgs, axis=-1)).to(self.device).float()
        self.right_images = torch.from_numpy(np.stack(right_imgs, axis=-1)).to(self.device).float()
        return True

    def plot_3d_points(self, x, y, z, color=None, title='Plot 3D of max correlation points'):
        if color is None:
            color = z
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.title.set_text(title)
        scatter = ax.scatter(x, y, z, c=color, cmap='viridis', marker='o')
        colorbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
        colorbar.set_label('Z Value Gradient')
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel('Z [mm]')
        ax.set_aspect('equal', adjustable='box')
        plt.show()

    def transform_gcs2ccs(self, points_3d, cam_name):
        xyz_gcs = torch.tensor(points_3d, dtype=torch.float32, device=self.device)
        k = torch.tensor(self.camera_params[cam_name]['kk'], dtype=torch.float32, device=self.device)
        r = torch.tensor(self.camera_params[cam_name]['r'], dtype=torch.float32, device=self.device)
        t = torch.tensor(self.camera_params[cam_name]['t'], dtype=torch.float32, device=self.device).view(3, 1)

        ones = torch.ones((xyz_gcs.shape[0], 1), dtype=torch.float32, device=self.device)
        xyz_h = torch.cat([xyz_gcs, ones], dim=1).T  # (4, N)

        rt = torch.cat([torch.cat([r, t], dim=1), torch.tensor([[0, 0, 0, 1]], dtype=torch.float32, device=self.device)], dim=0)
        xyz_cam = torch.matmul(rt, xyz_h)[:3, :]  # (3, N)

        x_norm = xyz_cam[:2, :] / (xyz_cam[2:3, :] + 1e-10)
        x_norm = torch.cat([x_norm, torch.ones((1, x_norm.shape[1]), dtype=torch.float32, device=self.device)], dim=0)

        uv = torch.matmul(k, x_norm)[:2, :]  # (2, N)
        return uv

    def points3d(self, x_lim, y_lim, z_lim, xy_step, z_step):
        x_vals = np.arange(x_lim[0], x_lim[1] + xy_step, xy_step)
        y_vals = np.arange(y_lim[0], y_lim[1] + xy_step, xy_step)
        z_vals = np.arange(z_lim[0], z_lim[1] + z_step, z_step)

        X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
        grid_np = np.stack((X, Y, Z), axis=-1)  # (Nx, Ny, Nz, 3)

        self.x_vals = torch.tensor(x_vals, dtype=torch.float32, device=self.device)
        self.y_vals = torch.tensor(y_vals, dtype=torch.float32, device=self.device)
        self.z_vals = torch.tensor(z_vals, dtype=torch.float32, device=self.device)

        self.grid = torch.tensor(grid_np, dtype=torch.float32, device=self.device)

    def bi_interpolation(self, images, uv_points, batch_size=100000):
        if images.ndim == 2:
            images = images.unsqueeze(-1)

        H, W, T = images.shape
        N = uv_points.shape[1]

        interpolated = torch.empty((N, T), dtype=torch.float32, device=self.device)
        std = torch.empty((N, T), dtype=torch.float32, device=self.device)

        for i in range(0, N, batch_size):
            end = min(i + batch_size, N)
            uv = uv_points[:, i:end]

            x = uv[0]
            y = uv[1]

            x1 = torch.clamp(torch.floor(x).long(), 0, W - 1)
            y1 = torch.clamp(torch.floor(y).long(), 0, H - 1)
            x2 = torch.clamp(x1 + 1, 0, W - 1)
            y2 = torch.clamp(y1 + 1, 0, H - 1)

            x_diff = (x - x1.float()).unsqueeze(1)
            y_diff = (y - y1.float()).unsqueeze(1)

            for t in range(T):
                p11 = images[y1, x1, t]
                p12 = images[y2, x1, t]
                p21 = images[y1, x2, t]
                p22 = images[y2, x2, t]

                interp = (
                    p11 * (1 - x_diff) * (1 - y_diff) +
                    p21 * x_diff * (1 - y_diff) +
                    p12 * (1 - x_diff) * y_diff +
                    p22 * x_diff * y_diff
                ).squeeze()

                interpolated[i:end, t] = interp
                std[i:end, t] = torch.std(torch.stack([p11, p12, p21, p22]), dim=0)

        return interpolated, std

    def get_kernel_indices(self, Kx=5, Ky=5, stride=1):
        Nx, Ny, Nz = self.grid.shape[:3]
        pad_x = Kx // 2
        pad_y = Ky // 2

        ix_centers = torch.arange(pad_x, Nx - pad_x, stride, device=self.device)
        iy_centers = torch.arange(pad_y, Ny - pad_y, stride, device=self.device)

        IX, IY = torch.meshgrid(ix_centers, iy_centers, indexing='ij')
        IX = IX.flatten()
        IY = IY.flatten()

        Nc = IX.shape[0]
        off_x = torch.arange(-pad_x, pad_x + 1, device=self.device)
        off_y = torch.arange(-pad_y, pad_y + 1, device=self.device)
        off_z = torch.arange(0, Nz, device=self.device)

        x_idx = IX[:, None] + off_x[None, :]
        y_idx = IY[:, None] + off_y[None, :]

        x_idx_full = x_idx[:, :, None, None]
        y_idx_full = y_idx[:, None, :, None]
        z_idx_full = off_z[None, None, None, :]

        self.grid_indices = torch.arange(Nx * Ny * Nz, device=self.device).reshape(Nx, Ny, Nz)
        kernels_idx = self.grid_indices[x_idx_full, y_idx_full, z_idx_full]

        return kernels_idx, (IX, IY)

    def spatial_temp_correl(self, interp_L_kernels, interp_R_kernels):
        Nc, Kx, Ky, Nz, T = interp_L_kernels.shape
        K = Kx * Ky * T

        L = interp_L_kernels.permute(0, 3, 1, 2, 4).reshape(Nc * Nz, K)
        R = interp_R_kernels.permute(0, 3, 1, 2, 4).reshape(Nc * Nz, K)

        L_mean = L.mean(dim=1, keepdim=True)
        R_mean = R.mean(dim=1, keepdim=True)

        Lz = L - L_mean
        Rz = R - R_mean

        num = (Lz * Rz).sum(dim=1)
        den = torch.sqrt((Lz ** 2).sum(dim=1) * (Rz ** 2).sum(dim=1) + 1e-10)

        corr_flat = num / den
        corr_all = corr_flat.view(Nc, Nz)

        corr_max, z_idx = corr_all.max(dim=1)
        z_best = self.z_vals[z_idx]

        return corr_all, corr_max, z_best

    def process(self, Kx=5, Ky=5, stride=1):
        Nx, Ny, Nz = self.grid.shape[:3]
        T = self.left_images.shape[2]
        grid_flat = self.grid.reshape(-1, 3)

        uv_left = self.transform_gcs2ccs(grid_flat, 'left')
        uv_right = self.transform_gcs2ccs(grid_flat, 'right')

        interp_L, std_L = self.bi_interpolation(self.left_images, uv_left)
        interp_R, std_R = self.bi_interpolation(self.right_images, uv_right)

        kernels_idx, (IX, IY) = self.get_kernel_indices(Kx=Kx, Ky=Ky, stride=stride)
        Nc = kernels_idx.shape[0]

        interp_L_k = interp_L[kernels_idx]
        interp_R_k = interp_R[kernels_idx]
        std_L_k = std_L[kernels_idx]
        std_R_k = std_R[kernels_idx]

        corr_all, corr_max, z_best = self.spatial_temp_correl(interp_L_k, interp_R_k)

        stdL_final = torch.mean(torch.std(std_L_k, dim=-1), dim=(1, 2))
        stdR_final = torch.mean(torch.std(std_R_k, dim=-1), dim=(1, 2))

        x_coords = self.x_vals[IX]
        y_coords = self.y_vals[IY]
        xyz_final = torch.stack([x_coords, y_coords, z_best], dim=1)

        return xyz_final, corr_max, corr_all, stdL_final, stdR_final
