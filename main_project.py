import cv2
import numpy as np
import os
import cupy as cp
import gc
import matplotlib.pyplot as plt
from pathlib import Path
from extras.debugger import plot_points_on_image, show_stereo_images_named
from include.SpatialCorrelation import StereoTemporalSpatialCorrel as StereoCorrelCupy
from include.SpatialCorrelation_pytorch import PyTorchStereoCorrel as StereoCorrelTorch
import extras.project_points as project_points

def calculate_and_plot_uv_differences(uv_points):
    """
    Calculate the differences between consecutive UV points, compute the average and standard deviation,
    and plot the differences.

    Parameters:
    ----------
    uv_points : np.ndarray
        Array of UV points with shape (2, N), where N is the number of points.

    Returns:
    -------
    avg_diff : float
        Average of the differences between consecutive UV points.
    std_diff : float
        Standard deviation of the differences between consecutive UV points.
    differences : np.ndarray
        Array of differences with shape (2, N-1).
    """
    
    # uv_points = cp.asnumpy(uv_points)  # Convert to NumPy array if using CuPy
    # Calculate differences between consecutive points
    differences = cp.diff(uv_points, axis=1)  # Shape: (2, N-1)

    # Compute the magnitude of the differences
    magnitudes = cp.linalg.norm(differences, axis=0) # Shape: (N-1,)
    # Calculate average and standard deviation
    avg_diff = cp.mean(magnitudes)
    std_diff = cp.std(magnitudes)
    magnitudes = cp.asnumpy(magnitudes)  # Convert to NumPy array if using CuPy
    avg_diff = cp.asnumpy(avg_diff)  # Convert to NumPy array if using CuPy
    std_diff = cp.asnumpy(std_diff)  # Convert to NumPy array if using CuPy
    # Plot the differences
    # plt.figure(figsize=(10, 6))
    # plt.plot(magnitudes, label='Differences Magnitude', marker='o')
    # plt.axhline(avg_diff, color='r', linestyle='--', label=f'Average: {avg_diff:.2f}')
    # plt.axhline(avg_diff + std_diff, color='g', linestyle='--', label=f'Avg + Std: {avg_diff + std_diff:.2f}')
    # plt.axhline(avg_diff - std_diff, color='g', linestyle='--', label=f'Avg - Std: {avg_diff - std_diff:.2f}')
    # plt.title('Differences Between Consecutive UV Points')
    # plt.xlabel('Point Index')
    # plt.ylabel('Difference Magnitude')
    # plt.legend()
    # plt.grid()
    # plt.show()

    return avg_diff, std_diff, differences

def read_images(path, images_list, n_imgs):       
        return [cv2.imread(os.path.join(path, str(img_name)), cv2.IMREAD_GRAYSCALE)
                    for img_name in images_list[0:n_imgs]]

def main():
    # Paths for yaml file and images
    yaml_file = 'cfg/SM4.yaml'
    # images_path = '/home/daniel/Insync/daniel.regner@labmetro.ufsc.br/Google Drive - Shared drives/VORIS - Media/Experimentos/SM3 - Padrão aleatório/2025 IMEKO - Imagens/20250513_1505_step10_plano_d2'
    images_path = '/home/daniel/Insync/daniel.regner@labmetro.ufsc.br/Google Drive - Shared drives/VORIS - Media/Experimentos/Sistemas Ativo - Congresso Metrologia 2025/20250722_2/correl/calota'

    left_imgs_list = sorted(os.listdir(os.path.join(images_path, 'left')))
    right_imgs_list = sorted(os.listdir(os.path.join(images_path, 'right')))
    # images_path = '/home/daniel/Pictures/sm3'
    n_img = 1
    # Determine XYZ bounds #(min, max)
    x_lim = (-100, 450) 
    y_lim = (-100, 300)
    z_lim = (-0, 10)
    dxyz = (25, 10) #xy step, z step

    zscan_cp = StereoCorrelCupy(yaml_file=yaml_file)
    zscan_torch = StereoCorrelTorch(yaml_file=yaml_file)
    # # Identify all images from path file
    left_imgs_cpu = read_images(path=os.path.join(images_path,'left'), images_list=left_imgs_list, n_imgs=n_img)
    right_imgs_cpu = read_images(path=os.path.join(images_path,'right'), images_list=right_imgs_list, n_imgs=n_img)

    zscan_cp.convert_images(left_imgs_cpu=left_imgs_cpu, right_imgs_cpu=right_imgs_cpu, apply_clahe=False, undist=True)
    zscan_torch.convert_images(left_imgs_cpu=left_imgs_cpu, right_imgs_cpu=right_imgs_cpu, apply_clahe=False, undist=True)


    zscan_cp.points3d(x_lim=x_lim, y_lim=y_lim, z_lim=z_lim, xy_step=dxyz[0], z_step=dxyz[1])
    zscan_torch.points3d(x_lim=x_lim, y_lim=y_lim, z_lim=z_lim, xy_step=dxyz[0], z_step=dxyz[1])
    grid_flat_cp = zscan_cp.grid.reshape(-1,3)
    uv_left_cp = zscan_cp.transform_gcs2ccs(points_3d=grid_flat_cp, cam_name='left')
    uv_right_cp = zscan_cp.transform_gcs2ccs(points_3d=grid_flat_cp, cam_name='right')

    print('cp uv_left', uv_left_cp.shape)
    print('cp uv_right', uv_right_cp.shape)

    grid_flat_torch = zscan_torch.grid.reshape(-1, 3)

    uv_left_torch, uv_l_mask = zscan_torch.transform_gcs2ccs(points_3d=grid_flat_torch, cam_name='left', image_shape=zscan_torch.left_images.shape[1:])
    uv_right_torch, uv_r_mask = zscan_torch.transform_gcs2ccs(points_3d=grid_flat_torch, cam_name='right', image_shape=zscan_torch.right_images.shape[1:])

    print('torch uv_left', uv_left_torch.shape)
    print('torch uv_right', uv_right_torch.shape)

    output_image_L = plot_points_on_image(image=left_imgs_cpu[0], points=cp.asnumpy(uv_left_cp), color=(0, 0, 255),
                                                   radius=3,
                                                   thickness=-1)
    output_image_R = plot_points_on_image(image=right_imgs_cpu[0], points=cp.asnumpy(uv_right_cp), color=(0, 255, 0),
                                                   radius=3,
                                                   thickness=-1)
    
    output_image_L = plot_points_on_image(image=output_image_L, points=uv_left_torch.cpu().numpy().T, color=(255, 0, 0), radius=7, thickness=1)
    output_image_R = plot_points_on_image(image=output_image_R, points=uv_right_torch.cpu().numpy().T, color=(255, 0, 0), radius=7, thickness=1)

    show_stereo_images_named(output_image_L, output_image_R, "Remaped points")
    cv2.waitKey(0)

    # Interpolate uv points

    inter_L_cp, _ = zscan_cp.bi_interpolation(uv_points=uv_left_cp, images=zscan_cp.left_images)
    inter_R_cp, _ = zscan_cp.bi_interpolation(uv_points=uv_right_cp, images=zscan_cp.right_images)

    print('inter_L_cp', inter_L_cp.shape)
    print('inter_R_cp', inter_R_cp.shape) 
    
    inter_L_torch = zscan_torch.interpolate_images(uv_points=uv_left_torch, images=zscan_torch.left_images, uv_mask=uv_l_mask)
    inter_R_torch = zscan_torch.interpolate_images(uv_points=uv_right_torch, images=zscan_torch.right_images, uv_mask=uv_r_mask)

    print('inter_L_torch', inter_L_torch.shape)
    print('inter_R_torch', inter_R_torch.shape)

    equal = np.allclose(cp.asnumpy(inter_L_cp), inter_L_torch.cpu().numpy(), atol=1e-10)
    print(f"Are the interpolated images equal? {equal}")




if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()