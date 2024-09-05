from doctest import debug
from pickletools import uint8

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

from numpy.ma.core import ones_like

import z_scan_temporal
import rectify_matrix
import debugger


def plot_3d_image(image):
    # Create x and y coordinates
    x = np.arange(0, image.shape[1])
    y = np.arange(0, image.shape[0])
    x, y = np.meshgrid(x, y)

    # Create the figure and 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    ax.plot_surface(x, y, image, cmap='gray')

    # Add labels and show the plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Intensity')
    plt.show()


def plot_3d_points(x, y, z):
    # Plot the 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(x, y, z, c=z, cmap='viridis', marker='o')
    ax.set_zlim(0, np.max(z))
    colorbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
    colorbar.set_label('Z Value Gradient')

    # Add labels
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_zlabel('Z [mm]')

    plt.show()


def points3d_cube(x_lim=(-5, 5), y_lim=(-5, 5), z_lim=(0, 5), xy_step=1.0, z_step=1.0, visualize=True):
    """
    Create a 3D space of combination from linear arrays of X Y Z
    Parameters:
        x_lim: Begin and end of linear space of X 
        x_lim: Begin and end of linear space of Y
        z_lim: Begin and end of linear space of Z
        xy_step: Step size between X and Y
        z_step: Step size between Z and X
        visualize: Visualize the 3D space
    Returns:
        cube_points: combination of X Y and Z
    """
    # Create x, y, z linear space
    x_lin = np.arange(x_lim[0], x_lim[1], step=xy_step)
    y_lin = np.arange(y_lim[0], y_lim[1], step=xy_step)
    z_lin = np.arange(z_lim[0], z_lim[1], step=z_step)

    # Combine all variables from x_lin, y_lin and z_lin
    mg1, mg2, mg3 = np.meshgrid(x_lin, y_lin, z_lin, indexing='ij')
    # Concatenate all vetors
    cube_points = np.stack([mg1, mg2, mg3], axis=-1).reshape(-1, 3)

    # Visualize space of points
    if visualize:
        plot_3d_points(x=cube_points[:, 0], y=cube_points[:, 1], z=cube_points[:, 2])

    return cube_points


def points2d_plane(xy=(-5, 5), xy_step=1.0, visualize=True):
    """
    Create a 3D space of combination from linear arrays of X Y Z
    Parameters:
        xy: Begin and end of linear space of X and Y
        xy_step: Step size between X and Y
        visualize: Visualize the 3D space
    Returns:
        cube_points: combination of X Y and Z
    """
    # Create x, y, z linear space
    x_lin = np.arange(xy[0], xy[1], step=xy_step)
    y_lin = np.arange(xy[0], xy[1], step=xy_step)

    # Combine all variables from x_lin, y_lin and z_lin
    mg1, mg2, mg3 = np.meshgrid(x_lin, y_lin, np.ones(x_lin.shape[0]), indexing='ij')
    # Concatenate all vetors
    cube_points = np.stack([mg1, mg2, mg3], axis=-1).reshape(-1, 3)

    # Visualize space of points
    if visualize:
        plot_3d_points(x=cube_points[:, 0], y=cube_points[:, 1], z=cube_points[:, 2])

    return cube_points


def gcs2ccs(xyz_gcs, k, dist, rot, tran):
    """
    Transform Global coordinate system to Camera coordinate system
    Parameters:
        xyz_gcs (array): Global coordinate system coordinates [X, Y, Z]
        k: instrinsic matrix
        dist: distortion vector [k1, k2, p1, p2, k3]
        rot: rotation matrix
        tran: translation vector
    Returns:
        uv_points: image points
    """
    xyz_gcs_1 = np.hstack((xyz_gcs, np.ones((xyz_gcs.shape[0], 1))))  # add one extra linhe of ones
    rt_matrix = np.vstack(
        (np.hstack((rot, tran[:, None])), [0, 0, 0, 1]))  # rot matrix and trans vector from gcs to ccs
    xyz_ccs = np.dot(rt_matrix, xyz_gcs_1.T)  # Multiply rotation and translation matrix to global points [X; Y; Z; 1]
    xyz_ccs_norm = np.hstack((xyz_ccs[:2, :].T / xyz_ccs[2, :, np.newaxis],
                              np.ones((xyz_ccs.shape[1], 1)))).T  # normalize vector [Xc/Zc; Yc/Zc; 1]
    xyz_ccs_norm_undist = undistorted_points(xyz_ccs_norm.T, dist)  # remove distortion from lens
    uv_points = np.dot(k, xyz_ccs_norm_undist.T)
    return uv_points


def undistorted_points(norm_points, distortion):
    """
    Remove distortion from normalized points.
    Parameters:
        norm_points: (N, 2) of (X, Y) normalized points
        distortion: [k1, k2, p1, p2, k3] distortions from camera
    Returns:
        undistorted_points: (N, 3) of (X, Y, 1) undistorted points
    """
    # radius of normalize points
    r2 = norm_points[:, 0] ** 2 + norm_points[:, 1] ** 2
    # distortion parameters
    k1, k2, p1, p2, k3 = distortion

    # Radial distortion correction
    factor = (1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3)
    x_corrected = norm_points[:, 0] * factor + 2 * p1 * norm_points[:, 0] * norm_points[:, 1] + p2 * (
            r2 + 2 * norm_points[:, 0] ** 2)
    y_corrected = norm_points[:, 1] * factor + p1 * (r2 + 2 * norm_points[:, 1] ** 2) + 2 * p2 * norm_points[:,
                                                                                                 0] * norm_points[:, 1]
    # return with extra columns of ones
    return np.hstack((np.stack([x_corrected, y_corrected], axis=-1), np.ones((norm_points.shape[0], 1))))


def plot_points_on_image(image, points, color=(0, 255, 0), radius=5, thickness=2):
    """
    Plot points on an image.

    Parameters:
    - image: The input image on which points will be plotted.
    - points: List of (u, v) coordinates to be plotted.
    - color: The color of the points (default: green).
    - radius: The radius of the circles to be drawn for each point.
    - thickness: The thickness of the circle outline.

    Returns:
    - output_image: The image with the plotted points.
    """
    # full_image = np.ones((np.max(points[:, 0]) + 1, np.max(points[:, 1]) + 1, 3), dtype=int)
    output_image = cv2.cvtColor(np.uint8(image), cv2.COLOR_GRAY2BGR)
    for (u, v, _) in points.T:
        # Ensure coordinates are within the image boundaries
        # if abs(u) > output_image.shape[0] and abs(v) > output_image.shape[1]:
        #     continue
        # else:
        # Draw a circle for each point on the image
        cv2.circle(output_image, (int(u), int(v)), radius, color, thickness)

    return output_image


def read_images(path, images_list):
    """
    Read all images from specified path.
    Parameters:
        path: (string) path to images folder.
        images_list: (list, string) list of images names.
    Return:
        images: (width, height, number images) array of image.
    """
    height, width = cv2.imread(os.path.join(path, str(images_list[0])), 0).shape
    images = np.zeros((height, width, len(images_list)), dtype=int)
    for n in range(len(images_list)):
        images[:, :, n] = cv2.imread(os.path.join(path, str(images_list[n])), cv2.IMREAD_GRAYSCALE)

    return images


def main():
    # Paths for yaml file and images
    yaml_file = 'cfg/20240828_bouget.yaml'
    images_path = 'images/SM3-20240828 - calib 10x10'

    # # Identify all images from path file
    left_images = read_images(os.path.join(images_path, 'left', ),
                              sorted(os.listdir(os.path.join(images_path, 'left'))))
    right_images = read_images(os.path.join(images_path, 'right', ),
                               sorted(os.listdir(os.path.join(images_path, 'right'))))

    # Read file containing all calibration parameters from stereo system
    Kl, Dl, Rl, Tl, Kr, Dr, Rr, Tr, R, T = rectify_matrix.load_camera_params(yaml_file=yaml_file)
    # xyz_points = z_scan_temporal.points3d_cube(xy=(-1, 1), z=(0, 1), xy_step=0.1, z_step=0.5, visualize=False)

    xy_points = points2d_plane(xy=(-300, 300), xy_step=10, visualize=False)
    uv_points_L = gcs2ccs(xy_points, Kl, Dl, Rl, Tl)
    uv_points_R = gcs2ccs(xy_points, Kr, Dr, Rr, Tr)
    output_image_L = plot_points_on_image(image=left_images[:, :, 11], points=uv_points_L, color=(0, 255, 0), radius=5,
                                          thickness=2)
    output_image_R = plot_points_on_image(image=right_images[:, :, 11], points=uv_points_R, color=(0, 255, 0), radius=5,
                                          thickness=2)

    debugger.show_stereo_images(output_image_L, output_image_R, "Remaped points")
    cv2.waitKey(0)
    # print('wait')


if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
