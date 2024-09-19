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





def points3d_cube(x_lim=(-5, 5), y_lim=(-5, 5), z_lim=(0, 5), xy_step=1.0, z_step=1.0, visualize=True):
    """
    Create a 3D space of combination from linear arrays of X Y Z
    Parameters:
        x_lim: Begin and end of linear space of X 
        y_lim: Begin and end of linear space of Y
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
        debugger.plot_3d_points(x=cube_points[:, 0], y=cube_points[:, 1], z=cube_points[:, 2])

    return cube_points

def points3d_cube_z(x_lim=(-5, 5), y_lim=(-5, 5), z_lim=(0, 5), xy_step=1.0, z_step=1.0, visualize=True):
    """
    Create a 3D space of combination from linear arrays of X Y Z
    Parameters:
        x_lim: Begin and end of linear space of X
        y_lim: Begin and end of linear space of Y
        z_lim: Begin and end of linear space of Z
        xy_step: Step size between X and Y
        z_step: Step size between Z and X
        visualize: Visualize the 3D space
    Returns:
        cube_points: combination of X Y and Z
    """
    # Create x, y, z linear space
    z_lin = np.arange(z_lim[0], z_lim[1], step=z_step)
    x_lin = np.zeros(z_lin.size)
    y_lin = np.zeros(z_lin.size)

    # Combine all variables from x_lin, y_lin and z_lin
    mg1, mg2, mg3 = np.meshgrid(x_lin, y_lin, z_lin, indexing='ij')
    # Concatenate all vetors
    cube_points = np.stack([mg1, mg2, mg3], axis=-1).reshape(-1, 3)

    # Visualize space of points
    if visualize:
        debugger.plot_3d_points(x=cube_points[:, 0], y=cube_points[:, 1], z=cube_points[:, 2])

    return cube_points

def points2d_plane(xy=(-5, 5), xy_step=1.0, visualize=True):
    """
    Create a 3D space of combination from linear arrays of X Y
    Parameters:
        xy: Begin and end of linear space of X and Y
        xy_step: Step size between X and Y
        visualize: Visualize the 3D space
    Returns:
        plane_points: combination of X Y from a defined Z
    """
    # Create x, y, z linear space
    x_lin = np.arange(xy[0], xy[1], step=xy_step)
    y_lin = np.arange(xy[0], xy[1], step=xy_step)

    # Combine all variables from x_lin, y_lin and z_lin
    mg1, mg2, mg3 = np.meshgrid(x_lin, y_lin, np.ones(x_lin.shape[0]), indexing='ij')
    # Concatenate all vetors
    plane_points = np.stack([mg1, mg2, mg3], axis=-1).reshape(-1, 3)

    # Visualize space of points
    if visualize:
        debugger.plot_3d_points(x=plane_points[:, 0], y=plane_points[:, 1], z=plane_points[:, 2])

    return plane_points


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


def filter_points_in_bounds(projected_points, image_width, image_height):
    """
    Filter points projected in bounds of image shape
    Parameters:
        projected_points: (N, 2) of (X, Y) projected points
        image_width: width of image
        image_height: height of image
    Returns:
        filtered_points: (N, 2) of (X, Y) filtered points
        valid_mask: boolean mask where valid points are True
    """
    # Extract u and v coordinates
    u, v = projected_points[0, :], projected_points[1, :]

    # Create a mask for points within the image boundaries
    valid_mask = (u >= 0) & (u < image_width) & (v >= 0) & (v < image_height)

    # Filter points using the valid mask
    filtered_points = projected_points[:, valid_mask]

    # Return the filtered points and the mask for valid indices
    return filtered_points, valid_mask

def gcs2ccs(xyz_gcs, k, dist, rot, tran):
    """
       Transform Global Coordinate System (GCS) to Camera Coordinate System (CCS).
       Parameters:
           xyz_gcs (array): Global coordinate system coordinates [X, Y, Z]
           k: Intrinsic matrix
           dist: Distortion vector [k1, k2, p1, p2, k3]
           rot: Rotation matrix
           tran: Translation vector
       Returns:
           uv_points: Image points
       """
    # add one extra linhe of ones
    xyz_gcs_1 = np.hstack((xyz_gcs, np.ones((xyz_gcs.shape[0], 1))))
    # rot matrix and trans vector from gcs to ccs
    rt_matrix = np.vstack(
        (np.hstack((rot, tran[:, None])), [0, 0, 0, 1]))
    # Multiply rotation and translation matrix to global points [X; Y; Z; 1]
    xyz_ccs = np.dot(rt_matrix, xyz_gcs_1.T)
    # Normalize by dividing by Z to get normalized image coordinates
    epsilon = 1e-10  # Small value to prevent division by zero
    xyz_ccs_norm = np.hstack((xyz_ccs[:2, :].T / np.maximum(xyz_ccs[2, :, np.newaxis], epsilon),
                              np.ones((xyz_ccs.shape[1], 1)))).T
    # remove distortion from lens
    xyz_ccs_norm_undist = undistorted_points(xyz_ccs_norm.T, dist)

    # Compute image's point as intrinsic K to XYZ CCS points normalized and undistorted
    uv_points = np.dot(k, xyz_ccs_norm_undist.T)
    return uv_points






def read_images(path, images_list, n_images):
    """
    Read all images from the specified path and stack them into a single array.
    Parameters:
        path: (string) path to images folder.
        images_list: (list of strings) list of image names.
    Returns:
        images: (height, width, number of images) array of images.
    """
    # Read all images using list comprehension
    images = [cv2.equalizeHist(cv2.imread(os.path.join(path, str(img_name)), cv2.IMREAD_GRAYSCALE)) for img_name in images_list[0:n_images]]

    # Convert list of images to a single 3D NumPy array
    images = np.stack(images, axis=-1).astype(np.uint8)  # Convert to uint8

    return images

def main():
    # Paths for yaml file and images
    yaml_file = 'cfg/20240918_bouget.yaml'
    images_path = '/home/daniel/Insync/daniel.regner@labmetro.ufsc.br/Google Drive - Shared drives/VORIS  - Equipe/Sistema de Medição 3 - Stereo Ativo - Projeção Laser/Imagens/Calibração/SM3-20240918 - calib 10x10'
    Nimg = 5
    # # Identify all images from path file
    left_images = read_images(os.path.join(images_path, 'left', ),
                              sorted(os.listdir(os.path.join(images_path, 'left'))), n_images=Nimg)
    right_images = read_images(os.path.join(images_path, 'right', ),
                               sorted(os.listdir(os.path.join(images_path, 'right'))), n_images=Nimg)

    # Read file containing all calibration parameters from stereo system
    Kl, Dl, Rl, Tl, Kr, Dr, Rr, Tr, R, T = rectify_matrix.load_camera_params(yaml_file=yaml_file)
    # xyz_points = z_scan_temporal.points3d_cube(xy=(-1, 1), z=(0, 1), xy_step=0.1, z_step=0.5, visualize=False)

    # xy_points = points2d_plane(xy=(-300, 300), xy_step=10, visualize=True)
    # xy_points = points3d_cube_z(x_lim=(-50, 50), y_lim=(-50, 50), z_lim=(-20, 20), xy_step=1, z_step=0.5, visualize=True)
    xy_points = points3d_cube(x_lim=(70, 100), y_lim=(80, 90), z_lim=(0, 1), xy_step=1, z_step=1, visualize=True)
    uv_points_L = gcs2ccs(xy_points, Kl, Dl, Rl, Tl)
    uv_points_R = gcs2ccs(xy_points, Kr, Dr, Rr, Tr)
    output_image_L = debugger.plot_points_on_image(image=left_images[:, :, 0], points=uv_points_L, color=(0, 255, 0), radius=5,
                                          thickness=2)
    output_image_R = debugger.plot_points_on_image(image=right_images[:, :, 0], points=uv_points_R, color=(0, 255, 0), radius=5,
                                          thickness=2)

    debugger.show_stereo_images(output_image_L, output_image_R, "Remaped points")
    cv2.waitKey(0)
    # print('wait')


if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
