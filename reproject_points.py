from pickletools import uint8

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

from numpy.ma.core import ones_like

import z_scan_temporal
import rectify_matrix


def gcs2ccs(xyz_gcs, k, dist, rot, tran):
    xyz_gcs_1 = np.hstack((xyz_gcs, np.ones((xyz_gcs.shape[0], 1))))  # add one extra linhe of ones
    rt_matrix = np.hstack((rot, tran[:, None]))  # rot matrix and trans vector from gcs to ccs
    xyz_ccs = np.dot(rt_matrix, xyz_gcs_1.T)
    xyz_ccs_norm = np.hstack((xyz_ccs[:2, :].T / xyz_ccs[2, :, np.newaxis],
                              np.ones((xyz_ccs.shape[1], 1)))).T
    xyz_ccs_norm_undist = undistorted_points(xyz_ccs_norm.T, dist)
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

def reproject_to_image_plane(XYZ_points, intrinsic_matrix, rotation_matrix, translation_vector):
    """
    Reproject 3D world coordinates to 2D image plane coordinates.

    Parameters:
    - XYZ_points: Array of 3D points in world coordinates (N x 3).
    - intrinsic_matrix: Camera intrinsic matrix (3 x 3).
    - rotation_matrix: Camera rotation matrix (3 x 3).
    - translation_vector: Camera translation vector (3 x 1).

    Returns:
    - uv_points: Array of 2D image plane coordinates (N x 2).
    """
    # Convert points to camera coordinate system
    # u = fx*Xc + cx*Zc -> u = fx*Xc/Zc + cx --> x' = Xc/Zc -> u = fx*x' + cx
    # v = fy*Yc + cy*zC -> u = fx*Yc/Zc + cy --> y' = Yc/Zc -> Yc= fy*y' + cy

    XYZ_camera = (rotation_matrix @ XYZ_points.T + translation_vector.reshape(3, -1)).T

    # Normalize by the Z component to get (X_c/Z_c, Y_c/Z_c, 1)
    X_c = XYZ_camera[:, 0]
    Y_c = XYZ_camera[:, 1]
    Z_c = XYZ_camera[:, 2]

    # Use the intrinsic matrix to project onto the image plane
    uv_points_homogeneous = intrinsic_matrix @ np.vstack((X_c / Z_c, Y_c / Z_c, np.ones_like(X_c)))

    # Convert from homogeneous to Cartesian coordinates
    uv_points = uv_points_homogeneous[:2, :].T

    return uv_points


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
    for (u, v) in points.reshape(-1, 2):
        # Ensure coordinates are within the image boundaries
        if abs(u) > output_image.shape[0]:
            u = 10
        if abs(v) > output_image.shape[1]:
            v = 10

        # Draw a circle for each point on the image
        cv2.circle(output_image, (int(u), int(v)), radius, color, thickness)

    return output_image


def main():
    # Paths for yaml file and images
    yaml_file = 'cfg/20240828_bouget.yaml'
    images_path = 'images/SM3-20240828 - calib 10x10'

    # # Identify all images from path file
    left_images = z_scan_temporal.read_images(os.path.join(images_path, 'left', ),
                                              sorted(os.listdir(os.path.join(images_path, 'left'))))

    # right_images = z_scan_temporal.read_images(os.path.join(images_path, 'right', ),
    #                                            sorted(os.listdir(os.path.join(images_path, 'right'))))

    # Read file containing all calibration parameters from stereo system
    Kl, Dl, Rl, Tl, _, Kr, Dr, Rr, Tr, _, R, T = rectify_matrix.load_camera_params(yaml_file=yaml_file)
    # xyz_points = z_scan_temporal.points3d_cube(xy=(-1, 1), z=(0, 1), xy_step=0.1, z_step=0.5, visualize=False)

    xy_points = z_scan_temporal.points2d_cube(xy=(-500,500), xy_step=10, visualize=False)
    uv_points = gcs2ccs(xy_points, Kl, Dr, Rl, Tl)
    output_image = plot_points_on_image(image=left_images[:, :, 11], points=uv_points, color=(0, 255, 0), radius=5,
                                        thickness=2)

    # uv_points = z_scan_temporal.project_points(xyz_points, Kl, Dl, Rl, Tl)
    # output_image = plot_points_on_image(left_images[:, :, 10], uv_points, color=(0, 255, 0), radius=5, thickness=2)
    cv2.namedWindow('output', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('output', 800, 600)
    cv2.imshow('output', output_image)
    cv2.waitKey(0)
    # print('wait')


if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
