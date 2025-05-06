import torch
import matplotlib.pyplot as plt
import os
import time
import numpy as np

from include.SpatialCorrelationTorch import SpatialCorrelatorTorch

def save_point_cloud(filename, xyz, corr=None, delimiter=','):
    if corr is not None:
        data = torch.cat((xyz, corr[:, None]), dim=1).cpu().numpy()
    else:
        data = xyz.cpu().numpy()
    np.savetxt(filename, data, delimiter=delimiter, header='x,y,z,corr' if corr is not None else 'x,y,z', comments='')
    print(f"Point cloud saved to {filename}")

def plot_3d_points(x, y, z, color=None, title='Plot 3D of max correlation points'):
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


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    yaml_file = 'cfg/SM3-20250424.yaml'
    images_path = '/home/daniel/Insync/daniel.regner@labmetro.ufsc.br/Google Drive - Shared drives/VORIS - Media/Experimentos/SM3 - Padrão aleatório/20250505  - Grasshopper f25 calotas'  # <- change this to your actual path
    left_imgs_list = sorted(os.listdir(os.path.join(images_path, 'left')))
    right_imgs_list = sorted(os.listdir(os.path.join(images_path, 'right')))

    n_imgs_v = [10, 15, 20, 25, 30]

    for n_img in n_imgs_v:
        print(f'\nNumber of images: {n_img}')
        t0 = time.time()

        Zscan = SpatialCorrelatorTorch(yaml_file=yaml_file, device=device)
        left_imgs = Zscan.read_images(path=os.path.join(images_path, 'left'), images_list=left_imgs_list, n_imgs=n_img)
        right_imgs = Zscan.read_images(path=os.path.join(images_path, 'right'), images_list=right_imgs_list, n_imgs=n_img)

        Zscan.convert_images(left_imgs, right_imgs, apply_clahe=True, undist=True)

        # Coarse scan
        Zscan.points3d(x_lim=(-180, 300), y_lim=(-140, 300), z_lim=(-500, 500), xy_step=20, z_step=1)
        xyz, corr, _, _ = Zscan.run_batch(r_xy=1, stride=2)
        mask = corr > 0.8
        filtered_xyz, filtered_corr = Zscan.filter_sparse_points(xyz[mask], corr[mask], min_neighbors=8, radius=60)

        print('1st Correlation time: {:.2f} s'.format(time.time() - t0))

        # Fine scan
        t1 = time.time()
        xlim = [filtered_xyz[:, 0].min().item(), filtered_xyz[:, 0].max().item()]
        ylim = [filtered_xyz[:, 1].min().item(), filtered_xyz[:, 1].max().item()]
        zlim = [filtered_xyz[:, 2].min().item(), filtered_xyz[:, 2].max().item()]
        print(f'X: {xlim[0]} to {xlim[1]} mm')
        print(f'Y: {ylim[0]} to {ylim[1]} mm')
        print(f'Z: {zlim[0]} to {zlim[1]} mm')

        Zscan.points3d(xlim, ylim, zlim, xy_step=1, z_step=0.5)
        xyz, corr, _, _ = Zscan.run_batch(r_xy=0.5, stride=2)
        mask = corr > 0.8
        filtered_xyz, filtered_corr = Zscan.filter_sparse_points(xyz[mask], corr[mask], min_neighbors=10, radius=5)

        print('2nd Correlation time: {:.2f} s'.format(time.time() - t1))
        print(f'Final Z: {filtered_xyz[:,2].min():.2f} to {filtered_xyz[:,2].max():.2f} mm')
        plot_3d_points(filtered_xyz[:, 0], filtered_xyz[:, 1], filtered_xyz[:, 2], color=filtered_corr, title='Filtered Points')

        # save_point_cloud(f'./sm3_calota_{n_img}_torch.csv', filtered_xyz, filtered_corr)

if __name__ == "__main__":
    main()
