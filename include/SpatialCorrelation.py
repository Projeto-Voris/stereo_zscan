import numpy as np
import cupy as cp
import yaml
import os
import matplotlib.pyplot as plt
import cv2
import gc
from scipy.spatial import cKDTree

class StereoTemporalSpatialCorrel:
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

    def estimate_batch_size(self, Kx, Ky, Nz, T, safety_margin=0.5):
        """
        Estima um batch_size seguro para a GPU, baseado no uso de memória por bloco.

        safety_margin: fração da GPU a ser utilizada (ex: 0.5 → usa até 50% da memória)
        """
        mem_total_GB = self.max_gpu_usage      # ex: 24.0 GB
        mem_target_GB = mem_total_GB * safety_margin  # ex: 12.0 GB

        bytes_per_float = 4  # float32
        mem_target_bytes = mem_target_GB * (1024 ** 3)

        # Memória por centro (Nc=1), L e R (2 blocos)
        mem_per_center = Kx * Ky * Nz * T * 2 * bytes_per_float

        # Batch size máximo
        batch_size = int(mem_target_bytes // mem_per_center)
        return max(1, batch_size)  # garante mínimo de 1

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
        uv_points_list = cp.empty((2, xyz_gcs.shape[0]), dtype=np.float32)

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
                (cp.hstack((rot, tran[:, None])), cp.array([0, 0, 0, 1], dtype=cp.float32))
            )

            # Multiply the RT matrix with global points [X; Y; Z; 1]
            xyz_ccs = cp.dot(rt_matrix, xyz_gcs_1.T)
            del xyz_gcs_1  # Immediately delete

            # Normalize by dividing by Z to get normalized image coordinates
            epsilon = 1e-10  # Small value to prevent division by zero
            xyz_ccs_norm = cp.hstack(
                (xyz_ccs[:2, :].T / cp.maximum(xyz_ccs[2, :, cp.newaxis], epsilon),
                 cp.ones((xyz_ccs.shape[1], 1), dtype=cp.float32))
            ).T
            del xyz_ccs  # Immediately delete

            
            # Compute image points using the intrinsic matrix K
            uv_points_batch = cp.dot(k, xyz_ccs_norm).astype(cp.float32)
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

    def bi_interpolation(self, images, uv_points, batch_size=100000):
        """
        Interpolação bilinear em batches para evitar OOM na GPU.

        Parameters:
            images : (H, W, T)
            uv_points : (2, N)
            batch_size : número máximo de pontos por batch

        Returns:
            interpolated : (N, T)
            std : (N, T)
        """
        images = cp.asarray(images)
        uv_points = cp.asarray(uv_points)

        if len(images.shape) == 2:
            images = images[:, :, cp.newaxis]

        height, width, num_images = images.shape
        N = uv_points.shape[1]

        interpolated = cp.empty((N, num_images), dtype=cp.float32)
        std = cp.empty((N, num_images), dtype=cp.float32)

        for i in range(0, N, batch_size):
            end = min(i + batch_size, N)
            uv_batch = uv_points[:, i:end]

            x = uv_batch[0].astype(cp.float32)
            y = uv_batch[1].astype(cp.float32)

            x1 = cp.clip(cp.floor(x).astype(cp.int32), 0, width - 1)
            y1 = cp.clip(cp.floor(y).astype(cp.int32), 0, height - 1)
            x2 = cp.clip(x1 + 1, 0, width - 1)
            y2 = cp.clip(y1 + 1, 0, height - 1)

            x_diff = x - x1
            y_diff = y - y1

            for k in range(num_images):
                p11 = images[y1, x1, k]
                p12 = images[y2, x1, k]
                p21 = images[y1, x2, k]
                p22 = images[y2, x2, k]

                interp = (
                    p11 * (1 - x_diff) * (1 - y_diff) +
                    p21 * x_diff * (1 - y_diff) +
                    p12 * (1 - x_diff) * y_diff +
                    p22 * x_diff * y_diff
                )

                std_dev = cp.std(cp.vstack([p11, p12, p21, p22]), axis=0)

                interpolated[i:end, k] = interp
                std[i:end, k] = std_dev

            del x1, x2, y1, y2, p11, p12, p21, p22
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()

        return interpolated, std

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

    def extract_kernels_fixed(self, Kx=5, Ky=5, stride=1):
        """
        Extrai kernels com tamanho fixo Kx × Ky ao redor de pontos centrais definidos com espaçamento (stride).

        Retorna:
            kernels: (N, Kx, Ky, Nz, 3)
            centers: lista de (x, y)
        """
        Nx, Ny, Nz, _ = self.grid.shape

        pad_x = Kx // 2
        pad_y = Ky // 2

        stride_ix = stride
        stride_iy = stride

        ix_centers = cp.arange(pad_x, Nx - pad_x, stride_ix)
        iy_centers = cp.arange(pad_y, Ny - pad_y, stride_iy)

        IX, IY = cp.meshgrid(ix_centers, iy_centers, indexing='ij')
        IX = IX.ravel()
        IY = IY.ravel()
        N_centers = IX.shape[0]

        # centros em mm
        centers = list(zip(cp.asnumpy(self.x_vals[IX]), cp.asnumpy(self.y_vals[IY])))

        # deslocamentos
        off_x = cp.arange(-pad_x, pad_x + 1)
        off_y = cp.arange(-pad_y, pad_y + 1)

        x_idx = IX[:, None] + off_x[None, :]
        y_idx = IY[:, None] + off_y[None, :]

        # Extrai (N, Kx, Ky, Nz, 3)
        kernels = self.grid[
            x_idx[:, :, None],
            y_idx[:, None, :],
            :,
            :
        ]

        return kernels, centers
    
    def get_kernel_indices(self, Kx=5, Ky=5, stride=1):
        """
        Gera os índices lineares (flattened) dos voxels que compõem cada kernel,
        sem replicar os valores 3D diretamente.

        Retorna:
            kernels_idx: (Nc, Kx, Ky, Nz) índices lineares dos voxels por kernel
            (IX, IY): coordenadas (i, j) dos centros
        """
        Nx, Ny, Nz = self.grid.shape[:3]
        pad_x = Kx // 2
        pad_y = Ky // 2

        # Índices válidos dos centros no plano XY (evitando bordas)
        ix_centers = np.arange(pad_x, Nx - pad_x, stride)
        iy_centers = np.arange(pad_y, Ny - pad_y, stride)

        IX, IY = np.meshgrid(ix_centers, iy_centers, indexing='ij')
        IX = IX.ravel()  # (Nc,)
        IY = IY.ravel()

        Nc = IX.size  # número total de centros

        # Offsets relativos do kernel
        off_x = np.arange(-pad_x, pad_x + 1)  # (Kx,)
        off_y = np.arange(-pad_y, pad_y + 1)  # (Ky,)
        off_z = np.arange(0, Nz)              # Kz = Nz (usa todo Z disponível)

        # Índices absolutos para cada centro
        x_idx = IX[:, None] + off_x[None, :]  # (Nc, Kx)
        y_idx = IY[:, None] + off_y[None, :]  # (Nc, Ky)

        # Meshgrid para acessar todos os pontos do kernel
        x_idx_full = x_idx[:, :, None, None]  # (Nc, Kx, 1, 1)
        y_idx_full = y_idx[:, None, :, None]  # (Nc, 1, Ky, 1)
        z_idx_full = off_z[None, None, None, :]  # (1, 1, 1, Kz)

        # Índices lineares no grid flatten (usado para mapeamento único)
        kernels_idx = self.grid_indices[
            x_idx_full,
            y_idx_full,
            z_idx_full
        ]  # shape: (Nc, Kx, Ky, Kz)

        return kernels_idx, (IX, IY)

    def spatial_temp_correl(self, interp_L_kernels, interp_R_kernels, batch_size=None):
        """
        Aplica correlação de Pearson entre blocos (Nc, Kx, Ky, Nz, T)
        usando todos os valores espaço-temporais (Kx  Ky  T) como vetor de entrada.

        Corrige o problema de T=1 mantendo a equação unificada para qualquer T.

        Retorna:
            corr_all : (Nc, Nz)
            corr_max : (Nc,)
            z_best   : (Nc,)
        """
        Nc, Kx, Ky, Nz, T = interp_L_kernels.shape
        K = Kx * Ky * T

        corr_parts = []

        for i in range(0, Nc, batch_size):
            end = min(i + batch_size, Nc)
            L_batch = interp_L_kernels[i:end]  # (B, Kx, Ky, Nz, T)
            R_batch = interp_R_kernels[i:end]

            B = L_batch.shape[0]

            # 1. Achata Kx × Ky × T em um vetor por voxel (B, K, Nz)
            L_flat = L_batch.transpose(0, 3, 1, 2, 4).reshape(B * Nz, K)
            R_flat = R_batch.transpose(0, 3, 1, 2, 4).reshape(B * Nz, K)

            # 2. Correlação de Pearson
            L_mu = cp.mean(L_flat, axis=1, keepdims=True)
            R_mu = cp.mean(R_flat, axis=1, keepdims=True)

            Lz = L_flat - L_mu
            Rz = R_flat - R_mu

            num = cp.sum(Lz * Rz, axis=1)
            den = cp.sqrt(cp.sum(Lz ** 2, axis=1) * cp.sum(Rz ** 2, axis=1))

            corr_flat = num / cp.maximum(den, 1e-10)     # (B * Nz,)
            corr_batch = corr_flat.reshape(B, Nz)        # (B, Nz)

            corr_parts.append(corr_batch)

            # Limpeza de memória
            del L_batch, R_batch, Lz, Rz, L_flat, R_flat
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()

        # 3. Concatenar todos os batches
        corr_all = cp.concatenate(corr_parts, axis=0)  # (Nc, Nz)
        corr_max = cp.nanmax(corr_all, axis=1)         # (Nc,)
        z_best_idx = cp.nanargmax(corr_all, axis=1)
        z_best = self.z_vals[z_best_idx]               # (Nc,)

        return corr_all, corr_max, z_best

    def process(self, Kx=5, Ky=5, stride=1, batch_size=None, y_offset=0):
        """
        Executa reconstrução estéreo espaço-temporal usando referência por índices de voxel.

        Usa projeção e interpolação uma única vez por ponto, evitando redundância.

        Retorna:
            xyz_final     (N, 3): centros com melhor Z
            corr_final    (N,): maior correlação por kernel
            corr_all      (Nc, Nz): correlação para cada voxel e profundidade
            stdL_final    (N,): desvio padrão médio da textura L
            stdR_final    (N,): desvio padrão médio da textura R
        """
        Nx, Ny, Nz = self.grid.shape[:3]
        T = self.left_images.shape[2]

        if batch_size is None:
            batch_size = self.estimate_batch_size(Kx, Ky, Nz, T, safety_margin=0.5)
            print(f"[INFO] Using estimated batch_size = {batch_size}")

        grid_flat = self.grid.reshape(-1, 3)  # (N, 3)
        self.grid_indices = np.arange(Nx * Ny * Nz).reshape(Nx, Ny, Nz)

        # 3. Projeção estéreo para pontos únicos
        uv_left = self.transform_gcs2ccs(grid_flat, cam_name='left')
        uv_right = self.transform_gcs2ccs(grid_flat, cam_name='right')

        # 4. Interpolação bilinear dos pontos únicos
        interp_L, std_L_map = self.bi_interpolation(self.left_images, uv_left)
        interp_R, std_R_map = self.bi_interpolation(self.right_images, uv_right)

        # 5. Construir mapas indexáveis (por voxel_id)
        kernels_idx, (IX, IY) = self.get_kernel_indices(Kx=Kx, Ky=Ky, stride=stride)  # (Nc, Kx, Ky, Nz)
        Nc, Kx, Ky, Nz = kernels_idx.shape


    # 4. Inicializa listas para acumular resultados
        xyz_parts = []
        corr_parts = []
        corr_all_parts = []
        stdL_parts = []
        stdR_parts = []

        # 5. Processa por batch de Nc
        for i in range(0, Nc, batch_size):
            end = min(i + batch_size, Nc)
            idx_range = slice(i, end)

            idx_kernels = kernels_idx[idx_range]  # (B, Kx, Ky, Nz)
            IX_batch = IX[idx_range]
            IY_batch = IY[idx_range]

            # Extrair blocos interpolados (B, Kx, Ky, Nz, T)
            interp_L_k = interp_L[idx_kernels]
            interp_R_k = interp_R[idx_kernels]
            std_L_k = std_L_map[idx_kernels]
            std_R_k = std_R_map[idx_kernels]
            # print('interp_L_k', interp_L_k.shape)
            assert interp_L_k.shape[3] == Nz
            # Correlação vetorizada por Z
            corr_all, corr_max, z_best = self.spatial_temp_correl(interp_L_k, interp_R_k, batch_size=batch_size)

            # Textura média por bloco
            stdL = cp.mean(cp.std(std_L_k, axis=-1), axis=(1, 2))
            stdR = cp.mean(cp.std(std_R_k, axis=-1), axis=(1, 2))

            # Coordenadas dos centros (x, y, z_best)
            x_coords = self.x_vals[IX_batch]
            y_coords = self.y_vals[IY_batch + y_offset]
            xyz = cp.stack([x_coords, y_coords, z_best], axis=1)

            # Acumula batch
            xyz_parts.append(xyz)
            corr_parts.append(corr_max)
            corr_all_parts.append(corr_all)
            stdL_parts.append(stdL)
            stdR_parts.append(stdR)

            # Libera memória
            del interp_L_k, interp_R_k, std_L_k, std_R_k
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()

        # 6. Concatenar todos os resultados
        xyz_final = cp.concatenate(xyz_parts, axis=0)
        corr_max = cp.concatenate(corr_parts, axis=0)
        corr_all = cp.concatenate(corr_all_parts, axis=0)
        stdL_final = cp.concatenate(stdL_parts, axis=0)
        stdR_final = cp.concatenate(stdR_parts, axis=0)

        return xyz_final, corr_max, corr_all, stdL_final, stdR_final
    

    def process_segmented_y(self, Kx=5, Ky=5, stride=1, batch_size=None, block_size_y=2, save_correlation=False):
        """
        Processa o grid 3D em fatias ao longo de Y para evitar estouro de memória GPU.
        """

        
        Ny_total = self.grid.shape[1]
        T = self.left_images.shape[2]



        xyz_all = []
        corr_all = []
        corrmap_all = []
        stdL_all = []
        stdR_all = []
        grid_backup = self.grid

        for y0 in range(0, Ny_total, block_size_y):
            y1 = min(y0 + block_size_y, Ny_total)

            # Salvar original

            # Fatia da grade
            self.grid = grid_backup[:, y0:y1, :, :]
            Nx, Ny, Nz = self.grid.shape[:3]
            self.grid_flat = self.grid.reshape(-1, 3)
            self.grid_indices = cp.arange(Nx * Ny * Nz).reshape(Nx, Ny, Nz)

            if batch_size is None:
                batch_size = self.estimate_batch_size(Kx, Ky, Nz, T, safety_margin=0.5)
                print(f"[INFO] Using estimated batch_size = {batch_size}")

            # Chama o pipeline normal
            xyz, corr, corrmap, stdL, stdR = self.process(
                Kx=Kx, Ky=Ky, stride=stride, batch_size=batch_size, save_correlation=save_correlation, y_offset=y0
            )

            # Acumula
            xyz_all.append(xyz)
            corr_all.append(corr)
            corrmap_all.append(corrmap)
            stdL_all.append(stdL)
            stdR_all.append(stdR)

            # Libera memória
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()


        # Concatena todos os resultados
        xyz_final = cp.concatenate(xyz_all, axis=0)
        corr_final = cp.concatenate(corr_all, axis=0)
        corrmap_final = cp.concatenate(corrmap_all, axis=0) #if save_correlation else None
        stdL_final = cp.concatenate(stdL_all, axis=0)
        stdR_final = cp.concatenate(stdR_all, axis=0)

        return xyz_final, corr_final, corrmap_final, stdL_final, stdR_final
    
    def process_segmented_z(self, Kx=5, Ky=5, stride=1, batch_size=None, Nz_block=50, save_correlation=False):
        """
        Segmenta o volume ao longo de Z para reduzir uso de memória.
        Reutiliza self.process() e acumula o melhor valor de correlação por voxel.

        Retorna:
            xyz_final     (Nc, 3)
            corr_final    (Nc,)
            corr_all      (Nc, Nz_total)
            stdL_final    (Nc,)
            stdR_final    (Nc,)
        """
        Nx, Ny, Nz_total = self.grid.shape[:3]
        self.grid_flat = self.grid.reshape(-1, 3)
        grid_backup = self.grid
        _, _, z_vals = self.x_vals, self.y_vals, self.z_vals

        # Pré-processa grid completo para saber centros
        self.grid_indices = np.arange(Nx * Ny * Nz_total).reshape(Nx, Ny, Nz_total)
        kernels_idx, (IX, IY) = self.get_kernel_indices(Kx=Kx, Ky=Ky, stride=stride)
        Nc = IX.shape[0]

        # Inicializa acumuladores
        corr_max = cp.full((Nc,), -cp.inf, dtype=cp.float32)
        z_best = cp.zeros((Nc,), dtype=cp.float32)
        stdL_all = []
        stdR_all = []
        corr_all_blocks = []

        for z0 in range(0, Nz_total, Nz_block):
            z1 = min(z0 + Nz_block, Nz_total)

            # Fatia grid ao longo de Z
            self.grid = grid_backup[:, :, z0:z1, :]
            self.z_vals = z_vals[z0:z1]

            print(f"[Z-SEGMENT] Processing z = {z0} to {z1} ({z1 - z0} slices)")

            # Atualiza auxiliares
            self.grid_flat = self.grid.reshape(-1, 3)
            self.grid_indices = np.arange(self.grid.size // 3).reshape(self.grid.shape[:3])

            # Executa correlação parcial
            xyz, corr, corr_block, stdL, stdR = self.process(Kx=Kx, Ky=Ky, stride=stride)

            if corr.shape[0] != Nc:
                raise ValueError("Número de centros mudou entre blocos de Z. Verifique stride, Kx/Ky e fatia.")

            if save_correlation:
                corr_all_blocks.append(corr_block)

            # Atualiza máximos
            improved = corr > corr_max
            corr_max[improved] = corr[improved]
            z_best[improved] = xyz[improved, 2]  # usa Z do centro local

            stdL_all.append(stdL)
            stdR_all.append(stdR)

            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()

        # Restaura grid original
        self.grid = grid_backup
        self.z_vals = z_vals

        # Reconstrói coordenadas finais (x, y) + melhor Z
        x_coords = self.x_vals[IX]
        y_coords = self.y_vals[IY]
        xyz_final = cp.stack([x_coords, y_coords, z_best], axis=1)

        stdL_final = 0 #cp.mean(cp.stack(stdL_all, axis=0), axis=0)
        stdR_final = 0 #cp.mean(cp.stack(stdR_all, axis=0), axis=0)
        corr_all = cp.concatenate(corr_all_blocks, axis=1) if save_correlation else None

        return xyz_final, corr_max, corr_all, stdL_final, stdR_final