import cv2
import numpy as np
import os
import cupy as cp
import gc
import matplotlib.pyplot as plt

from extras.debugger import plot_points_on_image, show_stereo_images_named
from include.InverseTriangulation import InverseTriangulation
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

def main():
    # Paths for yaml file and images
    yaml_file = 'cfg/SM4.yaml'
    # images_path = '/home/daniel/Insync/daniel.regner@labmetro.ufsc.br/Google Drive - Shared drives/VORIS - Media/Experimentos/SM3 - Padrão aleatório/2025 IMEKO - Imagens/20250513_1505_step10_plano_d2'
    images_path = 'fringe/calota/debug_images'

    left_imgs_list = sorted(os.listdir(os.path.join(images_path, 'left')))

    right_imgs_list = sorted(os.listdir(os.path.join(images_path, 'right')))
    # images_path = '/home/daniel/Pictures/sm3'
    n_img = 1
    # Determine XYZ bounds #(min, max)
    x_lim = (-50, 450) 
    y_lim = (-100, 300)
    z_lim = (-0, 10)
    dxyz = (25, 10) #xy step, z step

    Zscan = InverseTriangulation(yaml_file=yaml_file)
    # # Identify all images from path file
    left_imgs = Zscan.read_images(path=os.path.join(images_path,'left'), images_list=left_imgs_list, n_imgs=n_img)
    right_imgs = Zscan.read_images(path=os.path.join(images_path,'right'), images_list=right_imgs_list, n_imgs=n_img)
    Zscan.convert_images(left_imgs=left_imgs, right_imgs=right_imgs, apply_clahe=True, undist=True)

    points3d = Zscan.points3d(x_lim=x_lim, y_lim=y_lim, z_lim=z_lim, xy_step=dxyz[0], z_step=dxyz[1], visualize=True)
    print('points3d', points3d.shape)
    uv_left = Zscan.transform_gcs2ccs(points_3d=points3d, cam_name='left')
    uv_right = Zscan.transform_gcs2ccs(points_3d=points3d, cam_name='right')
    print('uv_left', uv_left.shape)
    print('uv_right', uv_right.shape)
    output_image_L = plot_points_on_image(image=cp.asnumpy(Zscan.left_images[:, :, 0]), points=uv_left, color=(0, 255, 0),
                                                   radius=4,
                                                   thickness=-1)
    output_image_R = plot_points_on_image(image=cp.asnumpy(Zscan.right_images[:, :, 0]), points=uv_right, color=(0, 255, 0),
                                                   radius=4,
                                                   thickness=-1)


    show_stereo_images_named(output_image_L, output_image_R, "Remaped points")
    cv2.waitKey(0)

    mask_left = (uv_left < 0) | (uv_left > left_imgs[0].shape[1])
    mask_right = (uv_right < 0) | (uv_right > right_imgs[0].shape[1])
    mask = np.any((mask_left | mask_right), axis=0)
    idx = np.where(mask)[0]
    out_pts = points3d[idx.get()]
    print(out_pts)
    # Calculate and plot differences for uv_left
    avg_diff_left, std_diff_left, differences_left = calculate_and_plot_uv_differences(uv_left)
    print(f"Left UV Points - Average Difference: {avg_diff_left} pixel/mm, Standard Deviation: {std_diff_left}")

    # Calculate and plot differences for uv_right
    avg_diff_right, std_diff_right, differences_right = calculate_and_plot_uv_differences(uv_right)
    print(f"Right UV Points - Average Difference: {avg_diff_right} pixel/mm, Standard Deviation: {std_diff_right}")
    print('wait')



if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()