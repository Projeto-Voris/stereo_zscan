import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import cupy as cp
import gc

import rectify_matrix
import debugger


def points3d_cube_gpu(x_lim=(-5, 5), y_lim=(-5, 5), z_lim=(0, 5), xy_step=1.0, z_step=1.0, visualize=True, max_memory_gb=4):
    """
    Create a 3D space of combinations from linear arrays of X, Y, Z on the GPU.
    Memory usage is limited to the specified max_memory_gb.

    Parameters:
        x_lim: Tuple (start, end) defining linear space of X
        y_lim: Tuple (start, end) defining linear space of Y
        z_lim: Tuple (start, end) defining linear space of Z
        xy_step: Step size between X and Y
        z_step: Step size between Z and X
        visualize: Boolean flag to visualize the 3D space
        max_memory_gb: Maximum GPU memory to use (in GB)

    Returns:
        cube_points: Combination of X, Y, and Z
    """

    # Calculate the size of the arrays
    x_lin = np.arange(x_lim[0], x_lim[1], step=xy_step)
    y_lin = np.arange(y_lim[0], y_lim[1], step=xy_step)
    z_lin = np.around(np.arange(z_lim[0], z_lim[1], step=z_step), decimals=2)

    # Calculate how many points we will have
    total_points = len(x_lin) * len(y_lin) * len(z_lin)
    bytes_per_float32 = 4  # 4 bytes per float32 number

    # Estimate the total memory needed for the final cube points (X, Y, Z, each as float32)
    total_memory_required = total_points * 3 * bytes_per_float32

    # Ensure it fits within max_memory_gb
    max_bytes = max_memory_gb * 1024**3  # Convert GB to bytes

    # Calculate how many points we can safely store in memory at once
    if total_memory_required > max_bytes:
        points_per_batch = int(max_bytes // (3 * bytes_per_float32))
        print(f"Total memory required exceeds limit. Processing in batches of {points_per_batch} points.")
    else:
        points_per_batch = total_points  # Process all at once

    # Initialize an empty list to store results (on the CPU)
    cube_points_list = []

    # Process points in batches
    for i in range(0, total_points, points_per_batch):
        # Create x, y, z linear spaces for the current batch
        x_batch = x_lin[i % len(x_lin)]
        y_batch = y_lin[i % len(y_lin)]
        z_batch = z_lin[i % len(z_lin)]

        # Create meshgrid on the GPU
        x_gpu = cp.arange(x_lim[0], x_lim[1], step=xy_step)
        y_gpu = cp.arange(y_lim[0], y_lim[1], step=xy_step)
        z_gpu = cp.arange(z_lim[0], z_lim[1], step=z_step)

        mg1, mg2, mg3 = cp.meshgrid(x_gpu, y_gpu, z_gpu, indexing='ij')

        # Concatenate vectors to form the cube points and bring back to CPU
        cube_points_batch = cp.stack([mg1, mg2, mg3], axis=-1).reshape(-1, 3)
        cube_points_list.append(cp.asnumpy(cube_points_batch))

        # Free GPU memory after processing each batch
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()

    # Combine all batches into a single array on CPU
    cube_points = np.vstack(cube_points_list)

    # Visualize if required
    if visualize:
        debugger.plot_3d_points(x=cube_points[:, 0], y=cube_points[:, 1], z=cube_points[:, 2])

    return cube_points


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
    z_lin = np.around(np.arange(z_lim[0], z_lim[1], step=z_step), decimals=2)

    # Combine all variables from x_lin, y_lin and z_lin
    mg1, mg2, mg3 = np.meshgrid(x_lin, y_lin, z_lin, indexing='ij')
    # Concatenate all vetors
    cube_points = np.stack([mg1, mg2, mg3], axis=-1).reshape(-1, 3)

    # Visualize space of points
    if visualize:
        debugger.plot_3d_points(x=cube_points[:, 0], y=cube_points[:, 1], z=cube_points[:, 2])

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
    factor = (1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3) #TODO: Fix radial distortion equation

    x_dist = norm_points[:, 0] * factor + 2 * p1 * norm_points[:, 0] * norm_points[:, 1] + p2 * (
            r2 + 2 * norm_points[:, 0] ** 2)
    y_dist = norm_points[:, 1] * factor + p1 * (r2 + 2 * norm_points[:, 1] ** 2) + 2 * p2 * norm_points[:,
                                                                                                 0] * norm_points[:, 1]
    # return with extra columns of ones
    return np.hstack((np.stack([x_dist, y_dist], axis=-1), np.ones((norm_points.shape[0], 1))))


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

    # introduce distortion from lens
    xyz_ccs_norm_dist = undistorted_points(xyz_ccs_norm.T, dist)

    # Compute image's point as intrinsic K to XYZ CCS points normalized and undistorted
    uv_points = np.dot(k, xyz_ccs_norm_dist.T)
    return uv_points


def gcs2ccs_gpu(xyz_gcs, k, dist, rot, tran, max_memory_gb=4):
    """
    Transform Global Coordinate System (GCS) to Camera Coordinate System (CCS) on the GPU,
    while limiting the GPU memory usage.

    Parameters:
        xyz_gcs (array): Global coordinate system coordinates [X, Y, Z]
        k: Intrinsic matrix
        dist: Distortion vector [k1, k2, p1, p2, k3]
        rot: Rotation matrix
        tran: Translation vector
        max_memory_gb: Maximum GPU memory usage in GB (default 4GB)

    Returns:
        uv_points: Image points (on the GPU, unless converted back to CPU).
    """

    # Convert all inputs to CuPy arrays for GPU computation
    xyz_gcs = cp.asarray(xyz_gcs)
    k = cp.asarray(k)
    dist = cp.asarray(dist)
    rot = cp.asarray(rot)
    tran = cp.asarray(tran)

    # Estimate the size of the input and output arrays
    num_points = xyz_gcs.shape[0]
    bytes_per_float32 = 8  # Simulate double-precision float usage

    # Estimate the memory required per point for transformation and intermediate steps
    memory_per_point = (4 * 3 * bytes_per_float32) + (3 * bytes_per_float32)  # For xyz_gcs_1 and xyz_ccs
    total_memory_required = num_points * memory_per_point

    # Maximum bytes allowed for memory usage
    max_bytes = max_memory_gb * 1024**3

    # Adjust the batch size based on memory limitations
    if total_memory_required > max_bytes:
        points_per_batch = int((max_bytes // memory_per_point) // 10)  # Reduce batch size more aggressively
        # print(f"Processing {points_per_batch} points per batch due to memory limitations.")
    else:
        points_per_batch = num_points  # Process all points at once

    # Initialize an empty list to store results (on the CPU)
    uv_points_list = []

    # Process points in batches
    for i in range(0, num_points, points_per_batch):
        end = min(i + points_per_batch, num_points)
        xyz_gcs_batch = xyz_gcs[i:end]

        # Debug: Check the shape of the batch
        # print(f"Processing batch {i // points_per_batch + 1}, size: {xyz_gcs_batch.shape}")

        # Add one extra line of ones to the global coordinates
        ones = cp.ones((xyz_gcs_batch.shape[0], 1), dtype=cp.float16)  # Double-precision floats
        xyz_gcs_1 = cp.hstack((xyz_gcs_batch, ones))

        # Create the rotation and translation matrix
        rt_matrix = cp.vstack(
            (cp.hstack((rot, tran[:, None])), cp.array([0, 0, 0, 1], dtype=cp.float16))
        )

        # Multiply the RT matrix with global points [X; Y; Z; 1]
        xyz_ccs = cp.dot(rt_matrix, xyz_gcs_1.T)
        del xyz_gcs_1  # Immediately delete

        # Normalize by dividing by Z to get normalized image coordinates
        epsilon = 1e-10  # Small value to prevent division by zero
        xyz_ccs_norm = cp.hstack(
            (xyz_ccs[:2, :].T / cp.maximum(xyz_ccs[2, :, cp.newaxis], epsilon),
             cp.ones((xyz_ccs.shape[1], 1), dtype=cp.float16))
        ).T
        del xyz_ccs  # Immediately delete

        # Apply distortion using the GPU
        xyz_ccs_norm_dist = undistorted_points_gpu(xyz_ccs_norm.T, dist)
        del xyz_ccs_norm  # Free memory

        # Compute image points using the intrinsic matrix K
        uv_points_batch = cp.dot(k, xyz_ccs_norm_dist.T)
        del xyz_ccs_norm_dist  # Free memory

        # Debug: Check the shape of the result
        # print(f"uv_points_batch shape: {uv_points_batch.shape}")

        # Transfer results back to CPU after processing each batch
        uv_points_list.append(cp.asnumpy(uv_points_batch))

        # Free GPU memory after processing each batch
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()

    # Ensure consistent dimensions when concatenating batches
    try:
        # Concatenate all batches along axis 0 (rows)
        uv_points = np.hstack(uv_points_list)  # Use np.hstack for matching shapes

    except ValueError as e:
        print(f"Error during concatenation: {e}")
        raise

    return uv_points


def undistorted_points_gpu(points, dist):
    """
    GPU version of the undistorted points function using CuPy.
    Applies radial and tangential distortion.

    Parameters:
        points: 2D array of normalized image coordinates [x, y].
        dist: Distortion coefficients [k1, k2, p1, p2, k3].

    Returns:
        Distorted points on the GPU.
    """
    # Extract distortion coefficients
    k1, k2, p1, p2, k3 = dist

    # Split points into x and y coordinates
    x, y = points[:, 0], points[:, 1]

    # Calculate r^2 (squared distance from the origin)
    r2 = x**2 + y**2

    # Radial distortion
    radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3

    # Tangential distortion
    x_tangential = 2 * p1 * x * y + p2 * (r2 + 2 * x**2)
    y_tangential = p1 * (r2 + 2 * y**2) + 2 * p2 * x * y

    # Compute distorted coordinates
    x_distorted = x * radial + x_tangential
    y_distorted = y * radial + y_tangential

    # Stack the distorted points
    distorted_points = cp.vstack([x_distorted, y_distorted, cp.ones_like(x)]).T

    # Clean up intermediate variables to free memory
    del x, y, r2, radial, x_tangential, y_tangential
    cp.get_default_memory_pool().free_all_blocks()

    return distorted_points


def read_images(path, images_list, n_images, visualize=False, CLAHE=False):
    """
    Read all images from the specified path and stack them into a single array.
    Parameters:
        path: (string) path to images folder.
        images_list: (list of strings) list of image names.
    Returns:
        images: (height, width, number of images) array of images.
    """
    # Read all images using list comprehension
    if CLAHE:
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        images = [clahe.apply(cv2.imread(os.path.join(path, str(img_name))), cv2.IMREAD_GRAYSCALE)
                  for img_name in images_list[0:n_images]]
    else:
        images = [cv2.imread(os.path.join(path, str(img_name)), cv2.IMREAD_GRAYSCALE)
                  for img_name in images_list[0:n_images]]

    # Convert list of images to a single 3D NumPy array
    images = np.stack(images, axis=-1).astype(np.uint8)  # Convert to uint8
    if visualize:
        for k in range(images.shape[2]):
            cv2.namedWindow(str(images_list[k]), cv2.WINDOW_NORMAL)
            cv2.resizeWindow(str(images_list[k]), 500, 500)
            cv2.imshow(str(images_list[k]), images[:, :, k])
            cv2.waitKey(0)
            cv2.destroyWindow(str(images_list[k]))

    return images


def main():
    # Paths for yaml file and images
    yaml_file = 'cfg/SM3_20240918_bouget.yaml'
    # images_path = 'images/SM4-20241004 -calib 25x25'
    images_path = '/home/daniel/Insync/daniel.regner@labmetro.ufsc.br/Google Drive - Shared drives/VORIS  - Equipe/Sistema de Medição 3 - Stereo Ativo - Projeção Laser/Imagens/Calibração/SM3-20240918 - calib 10x10'
    Nimg = 5
    # # Identify all images from path file
    left_images = read_images(os.path.join(images_path, 'left', ),
                              sorted(os.listdir(os.path.join(images_path, 'left'))), n_images=Nimg)
    right_images = read_images(os.path.join(images_path, 'right', ),
                               sorted(os.listdir(os.path.join(images_path, 'right'))), n_images=Nimg)

    # Read file containing all calibration qparameters from stereo system
    Kl, Dl, Rl, Tl, Kr, Dr, Rr, Tr, R, T = rectify_matrix.load_camera_params(yaml_file=yaml_file)
    # xyz_points = z_scan_temporal.points3d_cube(xy=(-1, 1), z=(0, 1), xy_step=0.1, z_step=0.5, visualize=False)

    xy_points = points3d_cube(x_lim=(-50, 250), y_lim=(-150, 200), z_lim=(-0, 100 ), xy_step=10, z_step=1,
                                             visualize=False)

    uv_points_L = gcs2ccs(xy_points, Kl, Dl, Rl, Tl)
    uv_points_R = gcs2ccs(xy_points, Kr, Dr, Rr, Tr)
    output_image_L = debugger.plot_points_on_image(image=left_images[:, :, 0], points=uv_points_L, color=(0, 255, 0),
                                                   radius=5,
                                                   thickness=1)
    output_image_R = debugger.plot_points_on_image(image=right_images[:, :, 0], points=uv_points_R, color=(0, 255, 0),
                                                   radius=5,
                                                   thickness=1)

    debugger.show_stereo_images(output_image_L, output_image_R, "Remaped points")
    cv2.waitKey(0)
    # print('wait')


if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
