import cv2
import os
from extras import debugger


def remap_rect_images(left, right, Kl, Dl, Rl, Pl, Kr, Dr, Rr, Pr):
    # Compute the undistortion and rectification transformation map
    map1x, map1y = cv2.initUndistortRectifyMap(Kl, Dl, Rl, Pl, (left.shape[1], left.shape[0]),
                                               cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(Kr, Dr, Rr, Pr, (right.shape[1], right.shape[0]),
                                               cv2.CV_32FC1)

    # Apply the rectification maps to the images
    rectified_left = cv2.remap(left, map1x, map1y, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(right, map2x, map2y, cv2.INTER_LINEAR)
    return rectified_left, rectified_right


def rectify_images(left, right, Kl, Dl, Kr, Dr, R, T, alpha_val=0):
    # Perform stereo rectification
    Rr1, Rr2, Pr1, Pr2, Q, ROI1, ROI2 = cv2.stereoRectify(Kl, Dl, Kr, Dr, (left.shape[1], left.shape[0]), R, T,
                                                          alpha=alpha_val)
    return Rr1, Rr2, Pr1, Pr2, Q, ROI1, ROI2


def load_camera_params(yaml_file):
    # Load the YAML file
    with open(yaml_file) as file:  # Replace with your file path
        params = yaml.safe_load(file)

        # Parse the matrices
    Kl = np.array(params['camera_matrix_left'], dtype=np.float64)
    Dl = np.array(params['dist_coeffs_left'], dtype=np.float64)
    Rl = np.array(params['rot_matrix_left'], dtype=np.float64)
    Tl = np.array(params['t_left'], dtype=np.float64)

    Kr = np.array(params['camera_matrix_right'], dtype=np.float64)
    Dr = np.array(params['dist_coeffs_right'], dtype=np.float64)
    Rr = np.array(params['rot_matrix_right'], dtype=np.float64)
    Tr = np.array(params['t_right'], dtype=np.float64)

    R = np.array(params['R'], dtype=np.float64)
    T = np.array(params['T'], dtype=np.float64)

    return Kl, Dl, Rl, Tl, Kr, Dr, Rr, Tr, R, T


import yaml
import numpy as np


def save_camera_parameters_to_yaml(file_path, camera_matrix_left, dist_coeffs_left, rot_matrix_left, proj_matrix_left,
                                   camera_matrix_right, dist_coeffs_right, rot_matrix_right, proj_matrix_right, R, T):
    """
    Save stereo camera parameters to a YAML file.

    Parameters:
        file_path (str): Path to the YAML file.
        camera_matrix_left (np.ndarray): Intrinsic parameters of the left camera.
        dist_coeffs_left (np.ndarray): Distortion coefficients of the left camera.
        proj_matrix_left (np.ndarray): Projection matrix of the left camera.
        camera_matrix_right (np.ndarray): Intrinsic parameters of the right camera.
        dist_coeffs_right (np.ndarray): Distortion coefficients of the right camera.
        proj_matrix_right (np.ndarray): Projection matrix of the right camera.
        R (np.ndarray): Rotation matrix between the two cameras.
        T (np.ndarray): Translation vector between the two cameras.
    """
    data = {
        'camera_matrix_left': camera_matrix_left.tolist(),
        'dist_coeffs_left': dist_coeffs_left.tolist(),
        'proj_matrix_left': proj_matrix_left.tolist(),
        'rot_matrix_left': rot_matrix_left.tolist(),
        'camera_matrix_right': camera_matrix_right.tolist(),
        'dist_coeffs_right': dist_coeffs_right.tolist(),
        'rot_matrix_right': rot_matrix_right.tolist(),
        'proj_matrix_right': proj_matrix_right.tolist(),
        'R': R.tolist(),
        'T': T.tolist(),
    }

    with open(file_path, 'w') as file:
        yaml.dump(data, file)

    print(f"Camera parameters saved to {file_path}")


def main():
    path = '../images/SM3-20240815_1'
    yaml_file = '../cfg/SM3_20240815_bouget.yaml'
    left_images = sorted(os.listdir(os.path.join(path, 'left')))
    right_images = sorted(os.listdir(os.path.join(path, 'right')))
    alpha = 1  # value of rectify map (0 - used ROI that are similar, 1 - uses all image)

    left_image = cv2.imread(os.path.join(path, 'left', left_images[0]), 0)
    right_image = cv2.imread(os.path.join(path, 'right', right_images[0]), 0)

    Kl, Dl, Rl, Pl, Kr, Dr, Rr, Pr, R, T = load_camera_params(yaml_file=yaml_file)
    mask_left, mask_right = debugger.mask_images(left_image, right_image, thres=180)
    debugger.show_stereo_images(left_image, right_image, 'mask')

    # left_image = cv2.bitwise_and(left_image, left_image, mask=mask_left)
    # right_image = cv2.bitwise_and(right_image, right_image, mask=mask_right)

    Rr1, Rr2, Pr1, Pr2, Q, ROI1, ROI2 = rectify_images(left_image, right_image, Kl=Kl, Dl=Dr, Kr=Kr, Dr=Dr,
                                                       R=R, T=T, alpha_val=alpha)

    rect_l, rect_r = remap_rect_images(left_image, right_image, Kl=Kl, Dl=Dr, Rl=Rr1, Pl=Pr1,
                                       Kr=Kr, Dr=Dr, Rr=Rr2, Pr=Pr2)

    rect_yaml_file = yaml_file.split('.yaml')[0] + '_rect_' + str(alpha) + '.yaml'

    save_camera_parameters_to_yaml(rect_yaml_file, camera_matrix_left=Kl, camera_matrix_right=Kr,
                                   dist_coeffs_left=Dl, dist_coeffs_right=Dr,
                                   rot_matrix_left=Rr1, rot_matrix_right=Rr2,
                                   proj_matrix_left=Pr1, proj_matrix_right=Pr2, R=R, T=T)

    debugger.show_stereo_images(rect_l, rect_r, name='Rectified images')


if __name__ == '__main__':
    main()
