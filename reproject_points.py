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
    rt_matrix = np.vstack((np.hstack((rot, tran[:, None])), [0, 0, 0, 1])) # rot matrix and trans vector from gcs to ccs
    xyz_ccs = np.dot(rt_matrix, xyz_gcs_1.T) # Multiply rotation and translation matrix to global points [X; Y; Z; 1]
    xyz_ccs_norm = np.hstack((xyz_ccs[:2, :].T / xyz_ccs[2, :, np.newaxis],
                              np.ones((xyz_ccs.shape[1], 1)))).T # normalize vector [Xc/Zc; Yc/Zc; 1]
    xyz_ccs_norm_undist = undistorted_points(xyz_ccs_norm.T, dist) # remove distortion from lens
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


def main():
    # Paths for yaml file and images
    yaml_file = 'cfg/20240828_bouget.yaml'
    images_path = 'images/SM3-20240828 - calib 10x10'

    # # Identify all images from path file
    left_images = z_scan_temporal.read_images(os.path.join(images_path, 'left', ),
                                              sorted(os.listdir(os.path.join(images_path, 'left'))))
    right_images = z_scan_temporal.read_images(os.path.join(images_path, 'right', ),
                                               sorted(os.listdir(os.path.join(images_path, 'right'))))

    # Read file containing all calibration parameters from stereo system
    Kl, Dl, Rl, Tl, Kr, Dr, Rr, Tr, R, T = rectify_matrix.load_camera_params(yaml_file=yaml_file)
    # xyz_points = z_scan_temporal.points3d_cube(xy=(-1, 1), z=(0, 1), xy_step=0.1, z_step=0.5, visualize=False)

    xy_points = z_scan_temporal.points2d_cube(xy=(-300,300), xy_step=10, visualize=False)
    uv_points_L = gcs2ccs(xy_points, Kl, Dl, Rl, Tl)
    uv_points_R = gcs2ccs(xy_points, Kr, Dr, Rr, Tr)
    output_image_L = plot_points_on_image(image=left_images[:,:,11], points=uv_points_L, color=(0, 255, 0), radius=5,
                                        thickness=2)
    output_image_R = plot_points_on_image(image=right_images[:,:,11], points=uv_points_R, color=(0, 255, 0), radius=5,
                                        thickness=2)

    debugger.show_stereo_images(output_image_L, output_image_R, "Remaped points")
    cv2.waitKey(0)
    # print('wait')


if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
