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
            'left': {'kk': cp.array([]), 'kc': cp.array([]), 'r': cp.array([]), 't': cp.array([])},
            'right': {'kk': cp.array([]), 'kc': cp.array([]), 'r': cp.array([]), 't': cp.array([])},
            'stereo': {'R': cp.array([]), 'T': cp.array([])}
        }
        self.read_yaml_file(yaml_file) # Carrega e converte para cp.float32

        self.max_gpu_usage_gb = self.set_datalimit() # Nome alterado para clareza
        # self.max_gpu_usage = self.set_datalimit() // 3 # Original, talvez muito conservador?

        self.x_vals = cp.array([])
        self.y_vals = cp.array([])
        self.z_vals = cp.array([])
        self.grid = cp.array([])
        # self.grid_indices = cp.array([]) # Removido, pois a lógica de kernel mudará

    def plot_3d_points(self, x, y, z, color=None, title='Plot 3D of max correlation points'):
        """
        Plot 3D points as scatter points where color is based on Z value
        Parameters:
            x: array of x positions
            y: array of y positions
            z: array of z positions
            color: Vector of point intensity grayscale
        """
        # Mover dados para CPU para plotting
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

        # Parse and convert matrices to CuPy float32 arrays
        self.camera_params['left']['kk'] = cp.array(params['camera_matrix_left'], dtype=cp.float32)
        self.camera_params['left']['kc'] = cp.array(params['dist_coeffs_left'], dtype=cp.float32)
        self.camera_params['left']['r'] = cp.array(params['rot_matrix_left'], dtype=cp.float32)
        self.camera_params['left']['t'] = cp.array(params['t_left'], dtype=cp.float32).reshape(3, 1) # Garantir que t seja (3,1)

        self.camera_params['right']['kk'] = cp.array(params['camera_matrix_right'], dtype=cp.float32)
        self.camera_params['right']['kc'] = cp.array(params['dist_coeffs_right'], dtype=cp.float32)
        self.camera_params['right']['r'] = cp.array(params['rot_matrix_right'], dtype=cp.float32)
        self.camera_params['right']['t'] = cp.array(params['t_right'], dtype=cp.float32).reshape(3, 1) # Garantir que t seja (3,1)

        self.camera_params['stereo']['R'] = cp.array(params['R'], dtype=cp.float32)
        self.camera_params['stereo']['T'] = cp.array(params['T'], dtype=cp.float32).reshape(3, 1)


    def read_images(self, path, images_list, n_imgs):
        images_cpu = [cv2.imread(os.path.join(path, str(img_name)), cv2.IMREAD_GRAYSCALE)
                      for img_name in images_list[0:n_imgs]]
        if not images_cpu or images_cpu[0] is None:
            raise FileNotFoundError(f"Could not read images from path: {path} with list: {images_list[0] if images_list else 'empty'}")
        return images_cpu # Retorna lista de arrays numpy

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
                # Nota: remove_img_distortion espera params da GPU, mas cv2.undistort opera em numpy.
                # Para eficiência, seria melhor undistort na GPU se possível, ou converter params para CPU aqui.
                # Por ora, vamos converter os params para numpy para cv2.undistort
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

        # Stack na CPU primeiro, depois transfere para GPU
        stacked_left_np = np.stack(processed_left_imgs, axis=-1).astype(np.uint8)
        stacked_right_np = np.stack(processed_right_imgs, axis=-1).astype(np.uint8)

        self.left_images = cp.asarray(stacked_left_np)
        self.right_images = cp.asarray(stacked_right_np)
        
        del processed_left_imgs, processed_right_imgs, stacked_left_np, stacked_right_np
        gc.collect()
        return True

    def remove_img_distortion(self, img_gpu, camera_name): # img_gpu deve ser cupy array
        # Esta função agora espera uma imagem GPU e params GPU.
        # No entanto, cv2.undistort não suporta CuPy arrays diretamente.
        # Solução: mover imagem para CPU, undistort, mover de volta. Ou usar OpenCv GpuMat se disponível e integrado.
        # Para simplificar e manter no CuPy o máximo possível, idealmente teríamos uma função de undistort na GPU.
        # Se não, o undistort deve ser feito na CPU antes de mover para self.left_images.
        # A lógica em convert_images foi ajustada para fazer undistort na CPU.
        # Esta função pode não ser mais necessária se convert_images cuidar de tudo.
        img_cpu = cp.asnumpy(img_gpu)
        k_cpu = cp.asnumpy(self.camera_params[camera_name]['kk'])
        kc_cpu = cp.asnumpy(self.camera_params[camera_name]['kc'])
        undistorted_img_cpu = cv2.undistort(img_cpu, k_cpu, kc_cpu)
        return cp.asarray(undistorted_img_cpu)


    def set_datalimit(self):
        device_id = 0
        cp.cuda.Device(device_id).use()
        total_memory = cp.cuda.runtime.getDeviceProperties(device_id)['totalGlobalMem']
        return total_memory / (1024 ** 3) # Retorna GB

    def estimate_batch_size(self, Kx, Ky, Nz, T, safety_margin=0.8): # Aumentei safety_margin
        # Usar max_gpu_usage_gb que é a memória total.
        # A estimativa deve considerar a memória *disponível*, não a total, ou aplicar uma margem maior.
        # Para este cálculo, usamos uma fração da memória total como alvo.
        mem_target_GB = self.max_gpu_usage_gb * safety_margin 
        
        bytes_per_float32 = 4
        mem_target_bytes = mem_target_GB * (1024 ** 3)

        # Memória para os dois arrays de kernel interpolados (L e R) que alimentam spatial_temp_correl
        # Forma: (Batch_Nc, Kx, Ky, Nz, T)
        mem_per_kernel_for_correlation_inputs = Kx * Ky * Nz * T * 2 * bytes_per_float32
        
        if mem_per_kernel_for_correlation_inputs == 0: # Evita divisão por zero
            return 1000 # Um valor padrão grande se não houver consumo por kernel

        batch_size_Nc = int(mem_target_bytes // mem_per_kernel_for_correlation_inputs)
        
        # Adicionar estimativa para outros arrays (voxels, UVs) se necessário,
        # mas os inputs da correlação costumam ser os maiores.
        # Ex: mem_voxels = Kx*Ky*Nz*3*2 (float16 para grid)
        # Ex: mem_uvs = Kx*Ky*Nz*2*4 (float32 para UVs) * 2 (L/R)
        # Ex: mem_interp_flat = Kx*Ky*Nz*T*4 (float32 para interp flat) * 2 (L/R)
        # A maior parte disso é sequencial, não simultânea com os inputs da correlação no pico.

        return max(1, batch_size_Nc)

    def transform_gcs2ccs(self, points_3d_gpu, cam_name): # points_3d_gpu já está na GPU
        # points_3d_gpu deve ser (N, 3) e float32
        k = self.camera_params[cam_name]['kk']      # (3,3) float32
        rot = self.camera_params[cam_name]['r']    # (3,3) float32
        tran = self.camera_params[cam_name]['t']   # (3,1) float32

        num_points = points_3d_gpu.shape[0]
        if num_points == 0:
            return cp.empty((2, 0), dtype=cp.float32)

        # Adicionar '1' à quarta coordenada: (N, 4)
        ones = cp.ones((num_points, 1), dtype=cp.float32)
        xyz_gcs_1 = cp.hstack((points_3d_gpu, ones)) # (N, 4)

        # Matriz de Rotação e Translação RT: (3, 4)
        rt_matrix = cp.hstack((rot, tran)) 

        # Transformar para coordenadas da câmera: (N, 3)
        # (RT @ xyz_gcs_1.T).T  -> (3,4) @ (4,N) = (3,N) -> (N,3)
        xyz_ccs_homogeneous = cp.dot(rt_matrix, xyz_gcs_1.T).T # (N,3)
        del xyz_gcs_1, rt_matrix, ones
        
        # Normalizar por Zc (terceira componente de xyz_ccs_homogeneous)
        zc = xyz_ccs_homogeneous[:, 2]
        epsilon = 1e-10
        # Evitar divisão por zero ou Zc muito pequeno/negativo (pontos atrás da câmera)
        valid_zc_mask = zc > epsilon 
        
        uv_points_cam = cp.empty((num_points, 2), dtype=cp.float32) # u,v
        
        if cp.any(valid_zc_mask):
            # Pontos normalizados (xn, yn): (N_valid, 2)
            xn = xyz_ccs_homogeneous[valid_zc_mask, 0] / zc[valid_zc_mask]
            yn = xyz_ccs_homogeneous[valid_zc_mask, 1] / zc[valid_zc_mask]

            # Aplicar matriz intrínseca K: ku*xn + cx, kv*yn + cy
            # k = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
            # u = fx * xn + cx
            # v = fy * yn + cy
            u_valid = k[0,0] * xn + k[0,2]
            v_valid = k[1,1] * yn + k[1,2]
            
            uv_points_cam[valid_zc_mask, 0] = u_valid
            uv_points_cam[valid_zc_mask, 1] = v_valid

        # Pontos inválidos (e.g., atrás da câmera) podem receber um valor sentinela, como -1
        uv_points_cam[~valid_zc_mask, :] = -1 
        
        del xyz_ccs_homogeneous, zc, valid_zc_mask
        # cp.get_default_memory_pool().free_all_blocks() # Menos frequente
        # gc.collect()
        return uv_points_cam.T # Retorna (2, N)

    def points3d(self, x_lim, y_lim, z_lim, xy_step, z_step):
        x_lin_np = np.arange(x_lim[0], x_lim[1] + xy_step, xy_step, dtype=np.float16)
        y_lin_np = np.arange(y_lim[0], y_lim[1] + xy_step, xy_step, dtype=np.float16)
        z_lin_np = np.arange(z_lim[0], z_lim[1] + z_step, z_step, dtype=np.float16)

        self.x_vals = cp.asarray(x_lin_np)
        self.y_vals = cp.asarray(y_lin_np)
        self.z_vals = cp.asarray(z_lin_np)

        # Gerar grid na CPU para economizar memória GPU se for gigantesco, depois transferir fatias
        # Ou gerar diretamente na GPU se soubermos que cabe (float16 ajuda)
        X, Y, Z = cp.meshgrid(self.x_vals, self.y_vals, self.z_vals, indexing='ij')
        self.grid = cp.stack((X, Y, Z), axis=-1)  # shape (Nx, Ny, Nz, 3), dtype=cp.float16
        del X, Y, Z
        gc.collect()


    def bi_interpolation(self, images_gpu, uv_points_gpu, batch_size_interp=2000000): # images (H,W,T), uv (2,N)
        if uv_points_gpu.shape[1] == 0:
            return cp.empty((0, images_gpu.shape[2]), dtype=cp.float32), \
                   cp.empty((0, images_gpu.shape[2]), dtype=cp.float32)

        images_gpu = cp.asarray(images_gpu) # Garantir que está na GPU
        uv_points_gpu = cp.asarray(uv_points_gpu)

        if len(images_gpu.shape) == 2: # Se for uma única imagem (H,W)
            images_gpu = images_gpu[:, :, cp.newaxis]

        height, width, num_images_T = images_gpu.shape
        N_total_points = uv_points_gpu.shape[1]

        interpolated_all = cp.empty((N_total_points, num_images_T), dtype=cp.float32)
        std_all = cp.empty((N_total_points, num_images_T), dtype=cp.float32)

        for i in range(0, N_total_points, batch_size_interp):
            end = min(i + batch_size_interp, N_total_points)
            uv_batch = uv_points_gpu[:, i:end] # (2, current_batch_N)

            x = uv_batch[0].astype(cp.float32) # Coordenadas u
            y = uv_batch[1].astype(cp.float32) # Coordenadas v

            # Clipping para garantir que os índices estejam dentro dos limites da imagem
            x = cp.clip(x, 0, width - 1 - epsilon) # Epsilon para evitar problemas no limite exato de width-1
            y = cp.clip(y, 0, height - 1 - epsilon)

            x1 = cp.floor(x).astype(cp.int32)
            y1 = cp.floor(y).astype(cp.int32)
            x2 = cp.clip(x1 + 1, 0, width - 1) # x1 pode ser width-1, então x1+1 pode ser width
            y2 = cp.clip(y1 + 1, 0, height - 1)
            
            # Garantir que x1 e x2 não sejam iguais se x estiver no limite
            # x1 = cp.minimum(x1, width - 2) # Se x1 é width-1, x2 se torna width-1. Clip x2.
            # y1 = cp.minimum(y1, height - 2)

            wa = (x - x1) # Pesos para interpolação
            wb = (y - y1)
            
            # (num_points_in_batch, num_images_T)
            current_batch_N_pts = x.shape[0]
            batch_interpolated_T = cp.empty((current_batch_N_pts, num_images_T), dtype=cp.float32)
            batch_std_T = cp.empty((current_batch_N_pts, num_images_T), dtype=cp.float32)

            for k_img_idx in range(num_images_T):
                img_slice_k = images_gpu[:, :, k_img_idx] # (H,W)
                
                p11 = img_slice_k[y1, x1] # (current_batch_N_pts,)
                p12 = img_slice_k[y2, x1]
                p21 = img_slice_k[y1, x2]
                p22 = img_slice_k[y2, x2]

                interp_val = p11 * (1 - wa) * (1 - wb) + \
                             p21 * wa * (1 - wb) + \
                             p12 * (1 - wa) * wb + \
                             p22 * wa * wb
                
                batch_interpolated_T[:, k_img_idx] = interp_val

                # std dos 4 vizinhos
                # Vstack na CPU seria lento. No CuPy:
                points_for_std = cp.stack((p11, p12, p21, p22), axis=0) # (4, current_batch_N_pts)
                batch_std_T[:, k_img_idx] = cp.std(points_for_std, axis=0)
                del p11, p12, p21, p22, points_for_std, interp_val

            interpolated_all[i:end, :] = batch_interpolated_T
            std_all[i:end, :] = batch_std_T
            
            del x, y, x1, y1, x2, y2, wa, wb, uv_batch, batch_interpolated_T, batch_std_T
            # cp.get_default_memory_pool().free_all_blocks() # Menos frequente aqui
            # gc.collect()
        
        # cp.get_default_memory_pool().free_all_blocks() # Ao final da função
        return interpolated_all, std_all


    def filter_sparse_points(self, xyz_gpu, corr_gpu, min_neighbors=5, radius=10):
        xyz_cpu = cp.asnumpy(xyz_gpu)
        corr_cpu = cp.asnumpy(corr_gpu)

        if xyz_cpu.shape[0] == 0:
            return cp.empty_like(xyz_gpu), cp.empty_like(corr_gpu)

        tree = cKDTree(xyz_cpu)
        neighbor_counts = tree.query_ball_point(xyz_cpu, r=radius)
        neighbor_counts = np.array([len(neighbors) for neighbors in neighbor_counts])
        dense_mask = neighbor_counts >= min_neighbors

        return cp.asarray(xyz_cpu[dense_mask]), cp.asarray(corr_cpu[dense_mask])
    
    # extract_kernels_fixed e get_kernel_indices podem não ser mais necessários da forma antiga
    # A lógica de extração de kernel agora está dentro de 'process'

    def spatial_temp_correl(self, interp_L_kernels, interp_R_kernels, batch_size_corr=None):
        # interp_L_kernels: (Current_Nc_Batch, Kx, Ky, Nz, T)
        Current_Nc_Batch, Kx, Ky, Nz, T = interp_L_kernels.shape
        
        if Current_Nc_Batch == 0:
             return cp.empty((0, Nz), dtype=cp.float32), \
                    cp.empty((0,), dtype=cp.float32), \
                    cp.empty((0,), dtype=cp.float32) # z_best_values

        # K é o tamanho do vetor de características por voxel (Kx * Ky * T)
        # No entanto, a correlação original é por voxel (ao longo de Z), usando um kernel 2D (Kx,Ky)
        # e todos os tempos T.
        # O reshape abaixo achata Kx*Ky*T em um único vetor por "posição Z" do kernel.
        # L_flat = L_batch.transpose(0, 3, 1, 2, 4).reshape(B * Nz, K)
        # Esta era a lógica antiga.
        # A correlação é feita para cada um dos `Nz` planos Z dentro de cada kernel.
        # Para cada kernel (indexado por Nc_idx) e para cada fatia Z (indexado por z_idx):
        #   Temos um patch L de (Kx, Ky, T) e um patch R de (Kx, Ky, T)
        #   Achatar Kx*Ky*T -> vetor V. Correlacionar V_L e V_R.
        # Resultado: corr_value(Nc_idx, z_idx)

        # K_feat = Kx * Ky * T # Tamanho do vetor de características
        # L_reshaped = interp_L_kernels.reshape(Current_Nc_Batch, Nz, K_feat) # (Nc, Nz, Kx*Ky*T)
        # R_reshaped = interp_R_kernels.reshape(Current_Nc_Batch, Nz, K_feat)

        # Revisitando a lógica original:
        # L_flat = L_batch.transpose(0, 3, 1, 2, 4).reshape(B * Nz, K)
        # L_batch era (B, Kx, Ky, Nz, T), onde B era um sublote de Nc.
        # Aqui, Current_Nc_Batch é o B.
        # Então, L_flat seria (Current_Nc_Batch * Nz, Kx*Ky*T)
        
        K_features = Kx * Ky * T
        if K_features == 0: # Evita erro se T=0 ou Kx/Ky=0
            corr_values = cp.zeros((Current_Nc_Batch, Nz), dtype=cp.float32)
        else:
            L_flat = interp_L_kernels.transpose(0, 3, 1, 2, 4).reshape(Current_Nc_Batch * Nz, K_features)
            R_flat = interp_R_kernels.transpose(0, 3, 1, 2, 4).reshape(Current_Nc_Batch * Nz, K_features)

            L_mu = cp.mean(L_flat, axis=1, keepdims=True)
            R_mu = cp.mean(R_flat, axis=1, keepdims=True)

            Lz = L_flat - L_mu
            Rz = R_flat - R_mu
            del L_mu, R_mu, L_flat, R_flat

            # Sum( (L_i - L_mu) * (R_i - R_mu) )
            numerator = cp.sum(Lz * Rz, axis=1)
            # Sqrt( Sum(L_i - L_mu)^2 * Sum(R_i - R_mu)^2 )
            denominator_L_sq_sum = cp.sum(Lz ** 2, axis=1)
            denominator_R_sq_sum = cp.sum(Rz ** 2, axis=1)
            del Lz, Rz
            
            denominator = cp.sqrt(denominator_L_sq_sum * denominator_R_sq_sum)
            del denominator_L_sq_sum, denominator_R_sq_sum

            corr_flat = numerator / cp.maximum(denominator, 1e-10) # (Current_Nc_Batch * Nz,)
            corr_values = corr_flat.reshape(Current_Nc_Batch, Nz) # (Current_Nc_Batch, Nz)
            del corr_flat, numerator, denominator
        
        corr_max_for_each_kernel = cp.nanmax(corr_values, axis=1) # (Current_Nc_Batch,)
        
        # Se todos forem NaN em alguma linha, argmax pode dar erro ou 0.
        # Lidar com kernels que só têm NaNs em corr_values[:, z_idx_slice]
        # cp.argmax pode ser usado se NaNs não forem um problema ou forem filtrados antes.
        # nanargmax ignora NaNs, que é o comportamento desejado.
        z_best_indices = cp.nanargmax(corr_values, axis=1) # (Current_Nc_Batch,)
        
        # self.z_vals são os valores Z globais (ou do slice Z atual se em process_segmented_z)
        z_best_actual_values = self.z_vals[z_best_indices] # (Current_Nc_Batch,)

        # gc.collect() # Menos frequente
        return corr_values, corr_max_for_each_kernel, z_best_actual_values

    def process(self, Kx=5, Ky=5, stride=1, nc_batch_size=None, y_coord_offset_val=0.0):
        # y_coord_offset_val: valor em mm a ser adicionado às coordenadas Y finais (usado por process_segmented_y)

        Nx, Ny_current_grid, Nz_current_grid = self.grid.shape[:3]
        T = self.left_images.shape[2]

        if nc_batch_size is None:
            nc_batch_size = self.estimate_batch_size(Kx, Ky, Nz_current_grid, T, safety_margin=0.35)
            print(f"[INFO] Using estimated Nc_batch_size = {nc_batch_size}")

        pad_x = Kx // 2
        pad_y = Ky // 2

        # Centros de kernel (índices no grid ATUAL)
        # Estes são para o self.grid como ele é (pode ser um slice de Y ou Z)
        ix_centers_all = cp.arange(pad_x, Nx - pad_x, stride, dtype=cp.int32)
        iy_centers_all = cp.arange(pad_y, Ny_current_grid - pad_y, stride, dtype=cp.int32)
        
        if len(ix_centers_all) == 0 or len(iy_centers_all) == 0 :
            print("[WARN] No valid kernel centers found. Check Kx, Ky, stride, and grid dimensions.")
            return cp.empty((0,3), dtype=cp.float32), cp.empty((0,), dtype=cp.float32), \
                   cp.empty((0,Nz_current_grid), dtype=cp.float32), \
                   cp.empty((0,), dtype=cp.float32), cp.empty((0,), dtype=cp.float32)

        IX_all, IY_all = cp.meshgrid(ix_centers_all, iy_centers_all, indexing='ij')
        IX_all = IX_all.ravel() # Índices X dos centros dos kernels
        IY_all = IY_all.ravel() # Índices Y dos centros dos kernels
        Nc_total = IX_all.shape[0]

        if Nc_total == 0:
            print("[WARN] Nc_total is 0. No kernels to process.")
            return cp.empty((0,3), dtype=cp.float32), cp.empty((0,), dtype=cp.float32), \
                   cp.empty((0,Nz_current_grid), dtype=cp.float32), \
                   cp.empty((0,), dtype=cp.float32), cp.empty((0,), dtype=cp.float32)


        # Listas para acumular resultados (transferir para CPU em lotes para evitar OOM na GPU para listas grandes)
        xyz_final_parts_cpu = []
        corr_max_parts_cpu = []
        corr_all_volume_parts_cpu = [] # (Nc_total, Nz_current_grid)
        stdL_final_parts_cpu = []
        stdR_final_parts_cpu = []

        # Offsets para construir todos os voxels de um kernel a partir de seu centro
        off_x_kernel = cp.arange(-pad_x, pad_x + 1, dtype=cp.int32) # (Kx,)
        off_y_kernel = cp.arange(-pad_y, pad_y + 1, dtype=cp.int32) # (Ky,)
        # off_z_kernel são todos os índices de Z: cp.arange(Nz_current_grid, dtype=cp.int32)

        print(f"[INFO] Total kernels to process (Nc_total): {Nc_total}. Batch size (nc_batch_size): {nc_batch_size}")

        for i in range(0, Nc_total, nc_batch_size):
            batch_start_time = cp.cuda.Event()
            batch_end_time = cp.cuda.Event()
            batch_start_time.record()

            current_batch_idx_start = i
            current_batch_idx_end = min(i + nc_batch_size, Nc_total)
            current_Nc_in_batch = current_batch_idx_end - current_batch_idx_start

            # Índices dos centros para o lote atual
            IX_batch_centers = IX_all[current_batch_idx_start:current_batch_idx_end] # (current_Nc_in_batch,)
            IY_batch_centers = IY_all[current_batch_idx_start:current_batch_idx_end]

            # 1. Gerar coordenadas 3D dos voxels para os kernels deste lote
            #   Resultado: kernel_voxel_coords (current_Nc_in_batch, Kx, Ky, Nz_current_grid, 3)
            
            #   kernel_abs_x_indices: (current_Nc_in_batch, Kx) = IX_batch_centers[:,None] + off_x_kernel[None,:]
            #   kernel_abs_y_indices: (current_Nc_in_batch, Ky) = IY_batch_centers[:,None] + off_y_kernel[None,:]
            #   kernel_abs_z_indices: (Nz_current_grid) = cp.arange(Nz_current_grid)
            
            #   Usando broadcasting para pegar os valores de self.grid:
            #   self.grid[
            #       kernel_abs_x_indices[:, :, cp.newaxis, cp.newaxis],  # (curr_Nc, Kx, 1, 1)
            #       kernel_abs_y_indices[:, cp.newaxis, :, cp.newaxis],  # (curr_Nc, 1, Ky, 1)
            #       kernel_abs_z_indices[cp.newaxis, cp.newaxis, cp.newaxis, :], # (1,1,1,Nz) -> Erro aqui, Z deve ser (curr_Nc,1,1,Nz) ou (Nz) se aplicado uniformemente
            #       : # Coordenadas XYZ
            #   ]
            #   Forma de self.grid: (Nx, Ny, Nz, 3)
            #   O stack é mais explícito:
            
            kernel_voxels_list = []
            for nc_idx in range(current_Nc_in_batch):
                center_ix = IX_batch_centers[nc_idx]
                center_iy = IY_batch_centers[nc_idx]
                
                # Índices absolutos para este kernel específico
                abs_x_indices_kernel = center_ix + off_x_kernel # (Kx,)
                abs_y_indices_kernel = center_iy + off_y_kernel # (Ky,)
                
                # Extrair o bloco (Kx, Ky, Nz_current_grid, 3) do grid
                # Precisa de meshgrid para os índices X, Y para indexação avançada correta
                # Ou fatiamento simples se os índices forem contíguos (stride=1 para kernel)
                # X_k_idx, Y_k_idx = cp.meshgrid(abs_x_indices_kernel, abs_y_indices_kernel, indexing='ij') # (Kx,Ky), (Kx,Ky)
                # current_kernel_voxels = self.grid[X_k_idx, Y_k_idx, :, :] # (Kx,Ky,Nz,3)
                # O acima é mais limpo se Kx,Ky são pequenos. Para Kx,Ky grandes, fatiamento é melhor:
                kernel_slice = self.grid[abs_x_indices_kernel[0]:abs_x_indices_kernel[-1]+1,
                                         abs_y_indices_kernel[0]:abs_y_indices_kernel[-1]+1,
                                         :, :]
                # Assegurar que o slice tenha Kx, Ky (pode ser menor nas bordas se não houver padding)
                # Assumimos que os centros já evitam bordas, então o slice é Kx,Ky,Nz,3
                kernel_voxels_list.append(kernel_slice)

            kernel_voxel_coords = cp.stack(kernel_voxels_list, axis=0).astype(cp.float32)
            del kernel_voxels_list
            # Forma: (current_Nc_in_batch, Kx, Ky, Nz_current_grid, 3)

            num_total_voxels_in_batch_kernels = current_Nc_in_batch * Kx * Ky * Nz_current_grid
            kernel_voxel_coords_flat = kernel_voxel_coords.reshape(num_total_voxels_in_batch_kernels, 3)
            del kernel_voxel_coords

            # 2. Projeção Estéreo
            uv_left_batch_flat = self.transform_gcs2ccs(kernel_voxel_coords_flat, cam_name='left') # (2, N_vox_batch)
            uv_right_batch_flat = self.transform_gcs2ccs(kernel_voxel_coords_flat, cam_name='right')
            del kernel_voxel_coords_flat

            # 3. Interpolação Bilinear
            interp_L_flat, std_L_val_flat = self.bi_interpolation(self.left_images, uv_left_batch_flat)
            interp_R_flat, std_R_val_flat = self.bi_interpolation(self.right_images, uv_right_batch_flat)
            del uv_left_batch_flat, uv_right_batch_flat

            # 4. Remodelar para Correlação: (current_Nc_in_batch, Kx, Ky, Nz_current_grid, T)
            interp_L_k = interp_L_flat.reshape(current_Nc_in_batch, Kx, Ky, Nz_current_grid, T)
            interp_R_k = interp_R_flat.reshape(current_Nc_in_batch, Kx, Ky, Nz_current_grid, T)
            del interp_L_flat, interp_R_flat
            
            # std_L_val_flat é (num_total_voxels_in_batch_kernels, T)
            # std_L_k_texture são os valores de std da interpolação, por voxel e por tempo T
            std_L_k_texture = std_L_val_flat.reshape(current_Nc_in_batch, Kx, Ky, Nz_current_grid, T)
            std_R_k_texture = std_R_val_flat.reshape(current_Nc_in_batch, Kx, Ky, Nz_current_grid, T)
            del std_L_val_flat, std_R_val_flat

            # 5. Correlação Espaço-Temporal
            # z_best_batch_values são os valores Z reais (não índices) de self.z_vals
            corr_all_plane_batch, corr_max_kernel_batch, z_best_batch_values = self.spatial_temp_correl(
                interp_L_k, interp_R_k
            ) # batch_size interno em spatial_temp_correl não é mais necessário aqui
            del interp_L_k, interp_R_k
            
            # 6. Calcular StdL e StdR para os kernels no z_best
            #    corr_all_plane_batch é (current_Nc_in_batch, Nz_current_grid)
            z_best_indices_in_Nz = cp.nanargmax(corr_all_plane_batch, axis=1) # (current_Nc_in_batch,)

            # std_L_k_texture é (current_Nc_in_batch, Kx, Ky, Nz_current_grid, T)
            # Queremos o std médio para a janela (Kx,Ky) no z_best, promediado sobre T.
            # Primeiro, selecione o std no z_best_idx para cada kernel no lote:
            # std_L_at_z_best_slice_KxKyT: (current_Nc_in_batch, Kx, Ky, T)
            # cp.take_along_axis requer que o array de índices tenha a mesma ndim que o array de dados.
            # Vamos iterar para selecionar, é mais simples de garantir a forma:
            
            selected_std_L_list = []
            selected_std_R_list = []
            for ker_idx_in_batch in range(current_Nc_in_batch):
                z_idx = z_best_indices_in_Nz[ker_idx_in_batch]
                selected_std_L_list.append(std_L_k_texture[ker_idx_in_batch, :, :, z_idx, :]) # (Kx,Ky,T)
                selected_std_R_list.append(std_R_k_texture[ker_idx_in_batch, :, :, z_idx, :])
            
            if selected_std_L_list:
                std_L_at_z_best_KxKyT = cp.stack(selected_std_L_list, axis=0) # (curr_Nc, Kx,Ky,T)
                std_R_at_z_best_KxKyT = cp.stack(selected_std_R_list, axis=0)
                
                # Agora promediar sobre Kx, Ky, e T para ter um valor por kernel
                mean_std_L_batch = cp.mean(std_L_at_z_best_KxKyT, axis=(1,2,3)) # (curr_Nc,)
                mean_std_R_batch = cp.mean(std_R_at_z_best_KxKyT, axis=(1,2,3))
                del std_L_at_z_best_KxKyT, std_R_at_z_best_KxKyT
            else: # Caso current_Nc_in_batch seja 0 (não deveria acontecer se Nc_total > 0)
                mean_std_L_batch = cp.empty((0,),dtype=cp.float32)
                mean_std_R_batch = cp.empty((0,),dtype=cp.float32)

            del std_L_k_texture, std_R_k_texture, selected_std_L_list, selected_std_R_list

            # 7. Coordenadas Finais para o lote
            #    x_coords e y_coords são do grid atual (que pode ser um slice)
            x_coords_batch = self.x_vals[IX_batch_centers] 
            y_coords_batch = self.y_vals[IY_batch_centers] + y_coord_offset_val # Adiciona offset em mm se houver
            
            xyz_batch_gpu = cp.stack([x_coords_batch, y_coords_batch, z_best_batch_values], axis=1)

            # Mover resultados do lote para CPU para liberar GPU
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


        if not xyz_final_parts_cpu: # Se nenhum lote foi processado
             print("[WARN] No data parts to concatenate. Returning empty arrays.")
             return cp.empty((0,3), dtype=cp.float32), cp.empty((0,), dtype=cp.float32), \
                   cp.empty((0,Nz_current_grid), dtype=cp.float32), \
                   cp.empty((0,), dtype=cp.float32), cp.empty((0,), dtype=cp.float32)

        # Concatenar todos os resultados (agora em CPU)
        xyz_final_all_cpu = np.concatenate(xyz_final_parts_cpu, axis=0)
        corr_max_all_cpu = np.concatenate(corr_max_parts_cpu, axis=0)
        corr_all_volume_final_cpu = np.concatenate(corr_all_volume_parts_cpu, axis=0)
        stdL_final_all_cpu = np.concatenate(stdL_final_parts_cpu, axis=0)
        stdR_final_all_cpu = np.concatenate(stdR_final_parts_cpu, axis=0)

        # Retornar como arrays CuPy
        return cp.asarray(xyz_final_all_cpu), cp.asarray(corr_max_all_cpu), \
               cp.asarray(corr_all_volume_final_cpu), \
               cp.asarray(stdL_final_all_cpu), cp.asarray(stdR_final_all_cpu)
    
    # --- Funções de Segmentação (precisam ser ajustadas para usar os novos retornos/parâmetros de process) ---
    # Nota: O parâmetro 'save_correlation' foi removido de process,
    #       pois corr_all_volume é sempre calculado e retornado.
    # Nota: O parâmetro 'y_offset' em process foi renomeado para 'y_coord_offset_val' e é um valor em mm.

    def process_segmented_y(self, Kx=5, Ky=5, stride=1, nc_batch_size_param=None, block_size_y_voxels=10): # block_size_y em número de voxels
        Ny_total = self.grid.shape[1]
        if Ny_total == 0: return # Ou retornar empty arrays
        
        xyz_all_parts = []
        corr_max_all_parts = []
        corrmap_all_parts = [] # Este é o corr_all_volume retornado por process
        stdL_all_parts = []
        stdR_all_parts = []
        
        grid_backup = self.grid
        y_vals_backup = self.y_vals # Guardar y_vals original (coordenadas em mm)

        for y0_idx in range(0, Ny_total, block_size_y_voxels):
            y1_idx = min(y0_idx + block_size_y_voxels, Ny_total)
            if y0_idx == y1_idx: continue # Skip empty slice

            print(f"[Y-SEGMENT] Processing Y-slice: indices {y0_idx} to {y1_idx-1}")

            self.grid = grid_backup[:, y0_idx:y1_idx, :, :]
            self.y_vals = y_vals_backup[y0_idx:y1_idx] # y_vals para o slice atual
            
            # O offset em mm a ser adicionado às coordenadas y calculadas no slice
            # Se y_vals_backup são os centros dos voxels, o offset é y_vals_backup[y0_idx]
            # Se y_vals_backup é np.arange(start, end, step), então y_vals_backup[0] é o início do grid global.
            # O y_coord_offset_val deve ser o valor em mm correspondente ao início do grid Y GLOBAL.
            # Não, ele deve ser o valor em mm do início DESTE SLICE.
            # Se self.y_vals[IY_batch_centers] dá o valor Y *dentro do slice*, 
            # então precisamos adicionar y_vals_backup[y0_idx] para torná-lo global.
            y_offset_mm_for_slice = y_vals_backup[y0_idx] if y0_idx > 0 else 0.0
            # No entanto, se y_vals[IY_batch_centers] já é global pq y_vals não foi fatiado, então 0.0
            # Minha implementação de process usa self.y_vals, que é fatiado.
            # Então, y_coords_batch = self.y_vals[IY_batch_centers] (local ao slice)
            # Precisamos adicionar y_vals_backup[y0_idx] (o valor real do início do slice Y)
            # O parâmetro y_coord_offset_val em process é para isso.

            xyz, corr_max, corrmap_slice, stdL, stdR = self.process(
                Kx=Kx, Ky=Ky, stride=stride, nc_batch_size=nc_batch_size_param,
                y_coord_offset_val=y_vals_backup[y0_idx] # Passa o valor absoluto em mm do início do slice
            )

            if xyz.shape[0] > 0: # Se encontrou resultados
                xyz_all_parts.append(xyz)
                corr_max_all_parts.append(corr_max)
                corrmap_all_parts.append(corrmap_slice) # (Nc_slice, Nz_slice)
                stdL_all_parts.append(stdL)
                stdR_all_parts.append(stdR)

            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()

        self.grid = grid_backup # Restaurar grid completo
        self.y_vals = y_vals_backup # Restaurar y_vals completo

        if not xyz_all_parts:
             return cp.empty((0,3)), cp.empty((0,)), cp.empty((0,self.grid.shape[2])), cp.empty((0,)), cp.empty((0,))

        xyz_final = cp.concatenate(xyz_all_parts, axis=0)
        corr_final = cp.concatenate(corr_max_all_parts, axis=0)
        # Concatenar corrmap_all_parts é mais complexo se Nc muda por slice.
        # Assumindo que o número de centros XZ é o mesmo para cada fatia Y.
        if corrmap_all_parts and all(c.shape[1] == corrmap_all_parts[0].shape[1] for c in corrmap_all_parts if c.ndim == 2 and c.size > 0): # Check Nz dim
            corrmap_final = cp.concatenate(corrmap_all_parts, axis=0)
        else: # Se as formas de Nz não baterem ou estiverem vazias
            Nz_total = self.grid.shape[2] if self.grid.ndim >=3 else 0
            num_total_points = xyz_final.shape[0]
            corrmap_final = cp.full((num_total_points, Nz_total), cp.nan, dtype=cp.float32) if num_total_points > 0 else cp.empty((0,Nz_total), dtype=cp.float32)

        stdL_final = cp.concatenate(stdL_all_parts, axis=0)
        stdR_final = cp.concatenate(stdR_all_parts, axis=0)

        return xyz_final, corr_final, corrmap_final, stdL_final, stdR_final


    def process_segmented_z(self, Kx=5, Ky=5, stride=1, nc_batch_size_param=None, Nz_block_voxels=50):
        Nx, Ny, Nz_total = self.grid.shape[:3]
        if Nz_total == 0: return # Ou retornar empty arrays
            
        # Gerar centros de kernel (IX, IY) para o grid XY completo uma vez.
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

        # Acumuladores para o melhor Z global
        # (Nc_for_xy_plane, Nz_total) para armazenar correlações de todos os blocos Z
        corr_map_overall_z = cp.full((Nc_for_xy_plane, Nz_total), -cp.inf, dtype=cp.float32)
        # stdL/R para o z_best global
        stdL_overall_best_z = cp.full((Nc_for_xy_plane,), cp.nan, dtype=cp.float32)
        stdR_overall_best_z = cp.full((Nc_for_xy_plane,), cp.nan, dtype=cp.float32)
        
        grid_backup = self.grid
        z_vals_backup = self.z_vals # Coordenadas Z em mm

        for z0_idx in range(0, Nz_total, Nz_block_voxels):
            z1_idx = min(z0_idx + Nz_block_voxels, Nz_total)
            if z0_idx == z1_idx: continue

            print(f"[Z-SEGMENT] Processing Z-slice: indices {z0_idx} to {z1_idx-1}")
            
            self.grid = grid_backup[:, :, z0_idx:z1_idx, :]
            self.z_vals = z_vals_backup[z0_idx:z1_idx] # z_vals para o slice atual
            current_Nz_in_slice = self.grid.shape[2]

            # Chamar process. Ele usará self.grid e self.z_vals fatiados.
            # y_coord_offset_val é 0.0 pois não estamos fatiando Y aqui.
            xyz_slice, corr_max_slice, corrmap_slice, stdL_slice, stdR_slice = self.process(
                Kx=Kx, Ky=Ky, stride=stride, nc_batch_size=nc_batch_size_param,
                y_coord_offset_val=0.0 
            )
            # xyz_slice: (Nc_xy, 3) onde Z é o melhor Z DENTRO DO SLICE
            # corr_max_slice: (Nc_xy,) correlação máxima DENTRO DO SLICE
            # corrmap_slice: (Nc_xy, current_Nz_in_slice) correlações para este slice Z
            # stdL_slice, stdR_slice: (Nc_xy,) std para o melhor Z DENTRO DO SLICE

            if xyz_slice.shape[0] == Nc_for_xy_plane: # Verificar se o número de centros é consistente
                # Colocar as correlações do slice no mapa geral
                corr_map_overall_z[:, z0_idx:z1_idx] = corrmap_slice
                
                # Atualizar stdL/R se a correlação deste slice for melhor que a global já registrada para aquele Z
                # Isso é um pouco mais complexo: precisamos saber qual era o Z global anterior e sua corr.
                # Alternativamente, podemos reconstruir stdL/R no final a partir do Z globalmente melhor.
            else:
                print(f"[WARN] Z-Slice {z0_idx}-{z1_idx}: Mismatch in Nc. Expected {Nc_for_xy_plane}, got {xyz_slice.shape[0]}. Skipping update for this slice.")

            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()

        self.grid = grid_backup # Restaurar grid completo
        self.z_vals = z_vals_backup # Restaurar z_vals completo

        # Agora, com corr_map_overall_z preenchido, encontrar o Z globalmente melhor
        corr_max_overall = cp.nanmax(corr_map_overall_z, axis=1) # (Nc_for_xy_plane,)
        z_best_indices_overall = cp.nanargmax(corr_map_overall_z, axis=1) # (Nc_for_xy_plane,)
        z_best_values_overall = self.z_vals[z_best_indices_overall] # (Nc_for_xy_plane,)

        # Coordenadas X, Y dos centros (são globais para o plano XY)
        x_coords_final = self.x_vals[IX_global_centers]
        y_coords_final = self.y_vals[IY_global_centers]
        
        xyz_final = cp.stack([x_coords_final, y_coords_final, z_best_values_overall], axis=1)

        # Reconstruir stdL/R para o z_best_overall.
        # Isso exigiria re-calcular ou armazenar std_L_k_texture de todos os blocos.
        # Por simplicidade, vamos deixar stdL/R como NaN ou 0 por enquanto,
        # ou o usuário pode adaptá-lo para buscar/recalcular o std no z_best_overall.
        # Para uma solução mais robusta, seria necessário armazenar stdL/stdR para cada (Nc,Nz) e depois selecionar.
        stdL_final = cp.full_like(corr_max_overall, cp.nan) 
        stdR_final = cp.full_like(corr_max_overall, cp.nan)
        # Se você tiver stdL_slice e stdR_slice para *cada voxel* de corrmap_slice,
        # você poderia fazer:
        # stdL_map_overall_z = cp.full((Nc_for_xy_plane, Nz_total), cp.nan, dtype=cp.float32)
        # ... popular stdL_map_overall_z similarmente a corr_map_overall_z ...
        # stdL_final = stdL_map_overall_z[cp.arange(Nc_for_xy_plane), z_best_indices_overall]

        return xyz_final, corr_max_overall, corr_map_overall_z, stdL_final, stdR_final

epsilon = 1e-6 # Pequeno valor para comparações de float e evitar divisão por zero em alguns contextos