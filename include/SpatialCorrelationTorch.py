import torch
import yaml
import cv2
import numpy as np
import os
from torchvision import transforms
from torch.nn.functional import grid_sample, layer_norm

class SpatialCorrelatorTorch:
    def __init__(self, yaml_file=None, device='cpu'):
        self.device = torch.device(device)
        self.camL = None
        self.camR = None
        if yaml_file is not None:
            self.read_yaml(yaml_file)
        self.grid_xyz = None

    def read_yaml(self, yaml_file):
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)

        self.KL = torch.tensor(data['camera_matrix_left'], dtype=torch.float32, device=self.device).reshape(3, 3)
        self.DL = torch.tensor(data['dist_coeffs_left'], dtype=torch.float32, device=self.device)
        self.RL = torch.tensor(data['rot_matrix_left'], dtype=torch.float32, device=self.device).reshape(3, 3)
        self.TL = torch.tensor(data['t_left'], dtype=torch.float32, device=self.device).reshape(3, 1)
        self.KR = torch.tensor(data['camera_matrix_right'], dtype=torch.float32, device=self.device).reshape(3, 3)
        self.DR = torch.tensor(data['dist_coeffs_right'], dtype=torch.float32, device=self.device)
        self.RR = torch.tensor(data['rot_matrix_right'], dtype=torch.float32, device=self.device).reshape(3, 3)
        self.TR = torch.tensor(data['t_right'], dtype=torch.float32, device=self.device).reshape(3, 1)
        self.R = torch.tensor(data['R'], dtype=torch.float32, device=self.device).reshape(3, 3)
        self.T = torch.tensor(data['T'], dtype=torch.float32, device=self.device).reshape(3, 1)

        # self.img_size = tuple(data['cam0']['resolution'][::-1])  # (W, H)
        # self._init_undistort_maps()

    def _init_undistort_maps(self):
        self.mapLx, self.mapLy = cv2.initUndistortRectifyMap(
            self.KL.cpu().numpy(), self.DL.cpu().numpy(), None, self.KL.cpu().numpy(),
            self.img_size, cv2.CV_32FC1)
        self.mapRx, self.mapRy = cv2.initUndistortRectifyMap(
            self.KR.cpu().numpy(), self.DR.cpu().numpy(), None, self.KR.cpu().numpy(),
            self.img_size, cv2.CV_32FC1)

    def read_images(self, path, images_list, n_imgs=None):
        if n_imgs:
            images_list = images_list[:n_imgs]
        imgs = []
        for fname in images_list:
            img = cv2.imread(os.path.join(path, fname), cv2.IMREAD_GRAYSCALE)
            self.img_size = img.shape[:2]
            imgs.append(img.astype(np.float32) / 255.0)

        self._init_undistort_maps()
        return np.stack(imgs, axis=0)

    def convert_images(self, left_stack, right_stack, apply_clahe=True, undist=True):
        if apply_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            left_stack = np.array([clahe.apply((img * 255).astype(np.uint8)) / 255.0 for img in left_stack])
            right_stack = np.array([clahe.apply((img * 255).astype(np.uint8)) / 255.0 for img in right_stack])

        if undist:
            left_stack = np.array([
                cv2.remap(img, self.mapLx, self.mapLy, interpolation=cv2.INTER_LINEAR)
                for img in left_stack
            ])
            right_stack = np.array([
                cv2.remap(img, self.mapRx, self.mapRy, interpolation=cv2.INTER_LINEAR)
                for img in right_stack
            ])

        # Convert to torch tensors [B, 1, H, W]
        self.left_stack = torch.from_numpy(left_stack).unsqueeze(1).to(self.device).float()
        self.right_stack = torch.from_numpy(right_stack).unsqueeze(1).to(self.device).float()

    def points3d(self, x_lim, y_lim, z_lim, xy_step, z_step):
        x = torch.arange(x_lim[0], x_lim[1]+1e-6, xy_step)
        y = torch.arange(y_lim[0], y_lim[1]+1e-6, xy_step)
        z = torch.arange(z_lim[0], z_lim[1]+1e-6, z_step)
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
        self.grid_xyz = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3).to(self.device)

    def run_batch(self, r_xy=1, stride=2):
        B, _, H, W = self.left_stack.shape
        patch_size = 2 * int(r_xy) + 1

        # Project 3D points
        ptsL_3D = (self.RL @ self.grid_xyz.T + self.TL).T
        ptsL = self.KL @ ptsL_3D.T
        ptsL = ptsL[:2] / ptsL[2:]
        ptsL = ptsL.T  # [N, 2]

        ptsR_3D = (self.RR @ self.grid_xyz.T + self.TR).T
        ptsR = self.KR @ ptsR_3D.T
        ptsR = ptsR[:2] / ptsR[2:]
        ptsR = ptsR.T  # [N, 2]

        # Normalize for grid_sample [-1, 1]
        norm_L = ptsL.clone()
        norm_R = ptsR.clone()
        norm_L[:, 0] = (norm_L[:, 0] / (W - 1)) * 2 - 1
        norm_L[:, 1] = (norm_L[:, 1] / (H - 1)) * 2 - 1
        norm_R[:, 0] = (norm_R[:, 0] / (W - 1)) * 2 - 1
        norm_R[:, 1] = (norm_R[:, 1] / (H - 1)) * 2 - 1

        N = self.grid_xyz.shape[0]
        grid_L = norm_L.view(N, 1, 1, 2)
        grid_R = norm_R.view(N, 1, 1, 2)

        grid_L = grid_L.view(1, -1, 1, 2).repeat(B, 1, 1, 1)
        grid_R = grid_R.view(1, -1, 1, 2).repeat(B, 1, 1, 1)


        patches_L = grid_sample(self.left_stack, grid_L, mode='bilinear', align_corners=True).squeeze(-1).squeeze(-1)
        patches_R = grid_sample(self.right_stack, grid_R, mode='bilinear', align_corners=True).squeeze(-1).squeeze(-1)

        patches_L = layer_norm(patches_L, (B,))
        patches_R = layer_norm(patches_R, (B,))
        corr = (patches_L * patches_R).sum(dim=0) / B  # [N]

        return self.grid_xyz, corr, ptsL, ptsR

    def filter_sparse_points(self, xyz, corr, min_neighbors=8, radius=60):
        from sklearn.neighbors import NearestNeighbors
        xyz_np = xyz.cpu().numpy()
        nbrs = NearestNeighbors(radius=radius).fit(xyz_np)
        n_neighbors = np.array([len(nbrs.radius_neighbors([p], return_distance=False)[0]) for p in xyz_np])
        mask = n_neighbors >= min_neighbors
        return xyz[mask], corr[mask]
