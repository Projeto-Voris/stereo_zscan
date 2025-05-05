import cv2
import numpy as np
import os
import yaml

def read_yaml_file(yaml_file, camera_params):
    """
    Read YAML file to extract cameras parameters
    """
    # Load the YAML file
    with open(yaml_file) as file:  # Replace with your file path
        params = yaml.safe_load(file)

        # Parse the matrices
    camera_params['left']['kk'] = np.array(params['camera_matrix_left'], dtype=np.float64)
    camera_params['left']['kc'] = np.array(params['dist_coeffs_left'], dtype=np.float64)
    camera_params['left']['r'] = np.array(params['rot_matrix_left'], dtype=np.float64)
    camera_params['left']['t'] = np.array(params['t_left'], dtype=np.float64)

    camera_params['right']['kk'] = np.array(params['camera_matrix_right'], dtype=np.float64)
    camera_params['right']['kc'] = np.array(params['dist_coeffs_right'], dtype=np.float64)
    camera_params['right']['r'] = np.array(params['rot_matrix_right'], dtype=np.float64)
    camera_params['right']['t'] = np.array(params['t_right'], dtype=np.float64)

    camera_params['stereo']['R'] = np.array(params['R'], dtype=np.float64)
    camera_params['stereo']['T'] = np.array(params['T'], dtype=np.float64)

    return camera_params


def rectify_and_save_images(left_image_path, right_image_path, output_dir, rectification_maps):
    """
    Rectify stereo images and save the rectified results.

    Parameters:
    ----------
    left_image_path : str
        Path to the left image.
    right_image_path : str
        Path to the right image.
    output_dir : str
        Directory to save the rectified images.
    rectification_maps : dict
        Dictionary containing rectification maps:
        {
            "left_map_x": np.ndarray,
            "left_map_y": np.ndarray,
            "right_map_x": np.ndarray,
            "right_map_y": np.ndarray
        }

    Returns:
    -------
    None
    """
    # Load the images
    left_image = cv2.imread(left_image_path, cv2.IMREAD_COLOR)
    right_image = cv2.imread(right_image_path, cv2.IMREAD_COLOR)

    if left_image is None or right_image is None:
        raise FileNotFoundError("One or both input images could not be loaded.")

    # Rectify the images using the rectification maps
    rectified_left = cv2.remap(left_image, rectification_maps["left_map1"], rectification_maps["left_map2"], cv2.INTER_LINEAR)
    rectified_right = cv2.remap(right_image, rectification_maps["right_map1"], rectification_maps["right_map2"], cv2.INTER_LINEAR)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'left'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'right'), exist_ok=True)

    # Save the rectified images
    left_output_path = os.path.join(output_dir,'left', f"rectified_{os.path.basename(left_image_path)}")
    right_output_path = os.path.join(output_dir,'right', f"rectified_{os.path.basename(right_image_path)}")
    cv2.imwrite(left_output_path, rectified_left)
    cv2.imwrite(right_output_path, rectified_right)

    # print(f"Rectified images saved to: {output_dir}")

def main():
    yaml_path = 'cfg/SM3-20250424.yaml'
    camera_params = {
            'left': {'kk': np.array([]), 'kc': np.array([]), 'r': np.array([]), 't': np.array([])},
            'right': {'kk': np.array([]), 'kc': np.array([]), 'r': np.array([]), 't': np.array([])},
            'stereo': {'R': np.array([]), 'T': np.array([])}
        }
    
    rectification_maps = {
        "left_map1": None,
        "left_map2": None,
        "right_map1": None,
        "right_map2": None
    }
    camera_params = read_yaml_file(yaml_path, camera_params)

    img_path = '/home/daniel/Insync/daniel.regner@labmetro.ufsc.br/Google Drive - Shared drives/VORIS - Media/Experimentos/SM3 - Padrão aleatório/20250425 -Grasshopper f_25mm'
    left_imgs_files= sorted(os.listdir(os.path.join(img_path, 'left')))
    right_imgs_files= sorted(os.listdir(os.path.join(img_path, 'right')))

    image_size = (2048, 2048)  # Ensure this matches the actual image resolution

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(cameraMatrix1=camera_params['left']['kk'], distCoeffs1=camera_params['left']['kc'],
                                          cameraMatrix2=camera_params['right']['kk'], distCoeffs2=camera_params['right']['kc'],
                                          imageSize=image_size, R=camera_params['stereo']['R'], T=camera_params['stereo']['T'], 
                                          flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    
    rectification_maps["left_map1"], rectification_maps["left_map2"] = cv2.initUndistortRectifyMap(camera_params['left']['kk'], camera_params['left']['kc'], R1, P1, (2048, 2048), cv2.CV_32FC1)
    rectification_maps["right_map1"], rectification_maps["right_map2"] = cv2.initUndistortRectifyMap(camera_params['right']['kk'], camera_params['right']['kc'], R2, P2, (2048, 2048), cv2.CV_32FC1)

    output_dir = os.path.join(img_path, 'rectified')
    os.makedirs(output_dir, exist_ok=True)
    for left_img_file, right_img_file in zip(left_imgs_files, right_imgs_files):
        left_image_path = os.path.join(img_path, 'left', left_img_file)
        right_image_path = os.path.join(img_path, 'right', right_img_file)
        rectify_and_save_images(left_image_path, right_image_path, output_dir, rectification_maps)
    print("Rectification completed for all images.")

if __name__ == "__main__":
    main()
