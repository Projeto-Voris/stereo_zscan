# Torch-based pipeline to perform temporal/spatial correlation on a 3D grid
# from stereo image pairs over time

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import yaml
import os
from extras import debugger


class TemporalStereoCorrelation:
    def __init__(self, camera_params, device='cuda'):
        self.device = torch.device(device)
        self.camera_params = camera_params
        self.left_images = None
        self.right_images = None

    def read_images(self, path, left_imgs, right_imgs, apply_undistort=True):
        def load_and_undistort(path, imgs, cam_key):
            kk = self.camera_params[cam_key]['kk']
            d = self.camera_params[cam_key]['kc']
            return [
                cv2.undistort(cv2.imread(os.path.join(path, str(img)), cv2.IMREAD_GRAYSCALE), kk, d) if apply_undistort
                else cv2.imread(os.path.join(path, str(img)), cv2.IMREAD_GRAYSCALE) for img in imgs
            ]

        left_path = os.path.join(path, 'left')
        right_path = os.path.join(path, 'right')
        left_imgs = load_and_undistort(left_path, left_imgs, 'left')
        right_imgs = load_and_undistort(right_path, right_imgs, 'right')

        self.left_images = torch.tensor(np.stack(left_imgs, axis=-1), dtype=torch.float32, device=self.device)
        self.right_images = torch.tensor(np.stack(right_imgs, axis=-1), dtype=torch.float32, device=self.device)

    def build_3d_grid(self, xlim=(0, 1), ylim=(0, 1), zlim=(0, 1), dxy=1, dz=1):
        x = torch.arange(xlim[0], xlim[1] + dxy, dxy, device=self.device)
        y = torch.arange(ylim[0], ylim[1] + dxy, dxy, device=self.device)
        z = torch.arange(zlim[0], zlim[1] + dz, dz, device=self.device)
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
        grid = torch.stack([X, Y, Z], dim=-1)  # shape (Nx, Ny, Nz, 3)
        return grid, x, y, z

    def extract_kernel_from_grid(self, global_grid, x_vals, y_vals, x0, y0, r_xy):
        dx = x_vals[1] - x_vals[0]
        dy = y_vals[1] - y_vals[0]
        ix = torch.argmin(torch.abs(x_vals - x0))
        iy = torch.argmin(torch.abs(y_vals - y0))
        r_ix = int((r_xy / dx).round().item())
        r_iy = int((r_xy / dy).round().item())
        return global_grid[ix - r_ix:ix + r_ix + 1, iy - r_iy:iy + r_iy + 1, :, :]  # (Kx, Ky, Nz, 3)

    def transform_to_uv(self, points, cam_key):
        if points.ndim > 2:
            points = points.reshape(-1, 3)

        k = torch.tensor(self.camera_params[cam_key]['kk'], dtype=torch.float32, device=self.device)
        r = torch.tensor(self.camera_params[cam_key]['r'], dtype=torch.float32, device=self.device)
        t = torch.tensor(self.camera_params[cam_key]['t'], dtype=torch.float32, device=self.device).view(3, 1)

        xyz = points.T  # (3, N)
        cam_coords = r @ xyz + t  # (3, N)
        x = cam_coords[0] / (cam_coords[2] + 1e-6)
        y = cam_coords[1] / (cam_coords[2] + 1e-6)

        uv = k @ torch.stack([x, y, torch.ones_like(x)], dim=0)
        return uv[:2].T  # (N, 2)

    def interpolate_images(self, images, uv):
        # images: (H, W, T), uv: (N, 2)
        H, W, T = images.shape
        images = images.permute(2, 0, 1).unsqueeze(0)  # (1, T, H, W)
        uv_norm = uv.clone()
        uv_norm[:, 0] = 2.0 * uv[:, 0] / (W - 1) - 1.0
        uv_norm[:, 1] = 2.0 * uv[:, 1] / (H - 1) - 1.0
        grid = uv_norm.view(1, -1, 1, 2)
        sampled = F.grid_sample(images, grid, mode='bilinear', align_corners=True)
        return sampled.squeeze(3).squeeze(0).T  # (N, T)

    def pearson_corr(self, left_vals, right_vals):
        left_mean = left_vals.mean(dim=1, keepdim=True)
        right_mean = right_vals.mean(dim=1, keepdim=True)
        num = ((left_vals - left_mean) * (right_vals - right_mean)).sum(dim=1)
        den = (left_vals - left_mean).pow(2).sum(dim=1).sqrt() * \
              (right_vals - right_mean).pow(2).sum(dim=1).sqrt()
        return num / (den + 1e-6)

    def correlate_patch(self, patch_3d):
        # patch_3d: (Kx, Ky, Kz, 3) → flatten to (N, 3)
        Kx, Ky, Kz, _ = patch_3d.shape
        points = patch_3d.reshape(-1, 3)
        uv_l = self.transform_to_uv(points, 'left')
        uv_r = self.transform_to_uv(points, 'right')

        I_l = self.interpolate_images(self.left_images, uv_l)
        I_r = self.interpolate_images(self.right_images, uv_r)

        corr = self.pearson_corr(I_l, I_r)
        corr = corr.view(Kx, Ky, Kz)
        best_z_idx = torch.argmax(corr, dim=2)
        return best_z_idx, corr


def read_yaml_file(yaml_file):
    """
    Read YAML file to extract cameras parameters
    """
    camera_params = {
        'left': {'kk': np.array([]), 'kc': np.array([]), 'r': np.array([]), 't': np.array([])},
        'right': {'kk': np.array([]), 'kc': np.array([]), 'r': np.array([]), 't': np.array([])},
        'stereo': {'R': np.array([]), 'T': np.array([])}
    }
    # Load the YAML file
    with open(yaml_file) as file:  # Replace with your file path
        params = yaml.safe_load(file)

        # Parse the matrices
    camera_params['left']['kk'] = np.array(params['camera_matrix_left'], dtype=np.float64)
    camera_params['left']['kc'] = np.array(params['dist_coeffs_left'], dtype=np.float64)
    camera_params['left']['r'] = np.array(params['rot_matrix_left'], dtype=np.float64)
    camera_params['left']['t'] = np.array(params['t_left'], dtype=np.float64)
    camera_params['right']['kk'] = np.array(params['camera_matrix_right'], dtype=np.float64)
    camera_params['right']['kc'] = np.array(params['dist_coeffs_right'], dtype=np.float64)
    camera_params['right']['r'] = np.array(params['rot_matrix_right'], dtype=np.float64)
    camera_params['right']['t'] = np.array(params['t_right'], dtype=np.float64)

    camera_params['stereo']['R'] = np.array(params['R'], dtype=np.float64)
    camera_params['stereo']['T'] = np.array(params['T'], dtype=np.float64)
    return camera_params


def main():
    yaml_file = '/home/daniel/PycharmProjects/stereo_zscan/cfg/20250212.yaml'
    images_path = '/home/daniel/PycharmProjects/stereo_zscan/images/20250423 - RRP'
    camera_params = read_yaml_file(yaml_file)
    correl = TemporalStereoCorrelation(camera_params=camera_params, device='cuda')
    correl.read_images(path=images_path, right_imgs=os.listdir(os.path.join(images_path, 'right')),
                       left_imgs=os.listdir(os.path.join(images_path, 'left')))
    grid, x, y, z = correl.build_3d_grid(xlim=(-10, 10), ylim=(-10, 10), zlim=(0, 20), dxy=0.05, dz=0.05)
    uv_l = correl.transform_to_uv(grid, 'left')
    uv_r = correl.transform_to_uv(grid, 'right')
    r_xy = 0.03
    results = []
    for x0 in x[::4]:  # step to downsample
        for y0 in y[::4]:
            patch = correl.extract_kernel_from_grid(grid, x, y, x0.item(), y0.item(), r_xy)

            best_z_idx, corr_volume = correl.correlate_patch(patch)
            Kx, Ky = best_z_idx.shape
            for i in range(Kx):
                for j in range(Ky):
                    z_idx = best_z_idx[i, j]
                    best_point = patch[i, j, z_idx]
                    corr_val = corr_volume[i, j, z_idx]
                    results.append((best_point.cpu().numpy(), corr_val.item()))


if __name__ == '__main__':
    main()
