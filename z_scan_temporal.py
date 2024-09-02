import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import rectify_matrix
from scipy.interpolate import RegularGridInterpolator as RGI


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

    ax.scatter(x, y, z, c='b', marker='o')

    # Add labels
    ax.set_xlabel('X Label (array1)')
    ax.set_ylabel('Y Label (array2)')
    ax.set_zlabel('Z Label (array3)')

    plt.show()


def points3d_cube(xy=(-5, 5), z=(0, 5), xy_step=1.0, z_step=1.0, visualize=True):
    """
    Create a 3D space of combination from linear arrays of X Y Z
    Parameters:
        xy: Begin and end of linear space of X and Y
        z: Begin and end of linear space of Z
        xy_step: Step size between X and Y
        z_step: Step size between Z and X
        visualize: Visualize the 3D space
    Returns:
        cube_points: combination of X Y and Z
    """
    # Create x, y, z linear space
    x_lin = np.arange(xy[0], xy[1], step=xy_step)
    y_lin = np.arange(xy[0], xy[1], step=xy_step)
    z_lin = np.arange(z[0], z[1], step=z_step)

    # Combine all variables from x_lin, y_lin and z_lin
    mg1, mg2, mg3 = np.meshgrid(x_lin, y_lin, z_lin, indexing='ij')
    # Concatenate all vetors
    cube_points = np.stack([mg1, mg2, mg3], axis=-1).reshape(-1, 3)

    # Visualize space of points
    if visualize:
        plot_3d_points(x=cube_points[:, 0], y=cube_points[:, 1], z=cube_points[:, 2])

    return cube_points


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


def project_points(points_ccs, K, dist, R, T):
    """
    Project points XYZ on image's plane
    Parameters:
        points_ccs: (N, 3) array
        K: (3,3) array of intrinsic parameters.
        dist: (1,5) array of distortion parameters.
        R: (3,3) array of rotation parameters - extrinsic parameters.
        T: (3,1) array of translation parameters - extrinsic parameters.
    Return:
        points: (N, 3) array of projected points.
    """
    # Construct projection matrix
    RT = np.vstack((np.hstack((R, T.reshape(-1, 1))), np.array([[0, 0, 0, 1]])))
    # Multiply RT to all points with one extra column
    points_gcs_hom = (RT @ np.hstack((points_ccs, np.ones((points_ccs.shape[0], 1)))).T).T
    # Normalize points x'=X/Z and y'=Y/Z
    points_norm_img = points_gcs_hom[:, :2] / points_gcs_hom[:, 2, np.newaxis]
    # remove distortions from lenses on points
    undistorted_pts = undistorted_points(points_norm_img, dist)
    # Return pixel coordinate on image plane from intrinsic paramenters
    return np.dot(K, undistorted_pts.T)


def interpolate_points(images, projected_points, method='linear'):
    """
    Interpolate points from images and projected points.

    Parameters:
    images: (height, width, num_images) array of images.
    projected_points: (3, N) array of projected points in homogeneous coordinates (u, v, 1).
    method: (string) interpolation method.

    Returns:
    inter_pts: (N, num_images) array of interpolated points.
    """
    # Remove the homogeneous coordinate and transpose to (N, 2) format
    projected_points_uv = projected_points[:2, :].T  # Shape: (N, 2)

    inter_pts = np.zeros((projected_points_uv.shape[0], images.shape[2]), dtype=float)
    # Define the image grid
    height, width, num_images = images.shape
    x = np.arange(width)
    y = np.arange(height)

    for n in range(num_images):
        # Create the interpolation function for each image
        interp = RGI((y, x), images[:, :, n], method=method, bounds_error=False, fill_value=None)

        # Perform interpolation for all projected points
        inter_pts[:, n] = interp(projected_points_uv)

    return inter_pts

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
        images[:, :, n] = cv2.imread(os.path.join(path, str(images_list[n])), 0)

    return images


def main():
    # Paths for yaml file and images
    yaml_file = 'cfg/20240815.yaml'
    images_path = 'images/SM3-20240820 - RRP'

    # Identify all images from path file
    left_images = read_images(os.path.join(images_path, 'left', ),
                              sorted(os.listdir(os.path.join(images_path, 'left'))))

    right_images = read_images(os.path.join(images_path, 'right', ),
                               sorted(os.listdir(os.path.join(images_path, 'right'))))

    # Read file containing all calibration parameters from stereo system
    Kl, Dl, Rl, Pl, Kr, Dr, Rr, Pr, R, T = rectify_matrix.load_camera_params(yaml_file=yaml_file)

    # Construct a 3D points (X,Y,Z) based on initial conditions and steps
    points_3d = points3d_cube(xy=(-1, 1), z=(0, 1), xy_step=0.5, z_step=0.5, visualize=False)
    projected_pts = project_points(points_ccs=points_3d, K=Kl, dist=Dl, R=Rl, T=T.T)
    inter_points = interpolate_points(left_images, projected_pts, method='linear')


if __name__ == '__main__':
    main()
