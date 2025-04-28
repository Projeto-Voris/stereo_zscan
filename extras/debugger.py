from sys import path_hooks

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import griddata
import cupy as cp

def plot_surf(uv_points, combined_std_mean):
    # Assuming UV points (uv_left) and combined_std_mean are given
    uv_points_np = cp.asnumpy(uv_points.T)  # Shape: (N, 2), convert to NumPy
    std_values_np = cp.asnumpy(combined_std_mean)  # Shape: (N,)

    # Plot the scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(uv_points_np[:, 0], uv_points_np[:, 1], c=std_values_np, cmap='viridis', s=10, alpha=0.8)
    plt.colorbar(label='Standard Deviation')
    plt.xlabel('U Coordinate')
    plt.ylabel('V Coordinate')
    plt.title('Scatter Plot of Standard Deviation at UV Points')
    plt.grid(True)
    plt.show()


def rectify_images(imgL, imgR, mtxL, distL, R1, P1, mtxR, distR, R2, P2):
    """Retifica imagens das câmeras esquerda e direita."""
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    left_map1, left_map2 = cv2.initUndistortRectifyMap(mtxL, distL, R1, P1, grayL.shape[::-1], cv2.CV_16SC2)
    right_map1, right_map2 = cv2.initUndistortRectifyMap(mtxR, distR, R2, P2, grayR.shape[::-1], cv2.CV_16SC2)
    rectifiedL = cv2.remap(grayL, left_map1, left_map2, cv2.INTER_LINEAR)
    rectifiedR = cv2.remap(grayR, right_map1, right_map2, cv2.INTER_LINEAR)
    return rectifiedL, rectifiedR


def calculate_disparity(rectifiedL, rectifiedR):
    """Calcula a disparidade entre as imagens retificadas."""
    stereo = cv2.StereoBM_create(numDisparities=16 * 10, blockSize=15)
    disparity = stereo.compute(rectifiedL, rectifiedR)
    return disparity


def generate_point_cloud(disparity, Q, imgL):
    """Gera a nuvem de pontos 3D a partir da disparidade."""
    points_3D = cv2.reprojectImageTo3D(disparity, Q)
    mask = disparity > disparity.min()
    output_points = points_3D[mask]
    output_colors = imgL[mask]
    return output_points, output_colors


def save_point_cloud(points, colors, output_file, scale_factor=0.00345):
    """Salva a nuvem de pontos em um arquivo PLY."""
    points_scaled = points * scale_factor
    ply_header = '''ply
                    format ascii 1.0
                    element vertex %(vert_num)d
                    property float x
                    property float y
                    property float z
                    property uchar red
                    property uchar green
                    property uchar blue
                    end_header
                    '''
    with open(output_file, 'w') as f:
        f.write(ply_header % dict(vert_num=len(points_scaled)))
        for point, color in zip(points_scaled, colors):
            f.write(f"{point[0]} {point[1]} {point[2]} {color[2]} {color[1]} {color[0]}\n")


def show_image(image):
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyWindow('Image')

def show_stereo_images(imgR, imgL):
    img_concatenate = np.concatenate((imgL, imgR), axis=1)
    cv2.namedWindow('Stereo', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Stereo', int(img_concatenate.shape[1] / 4), int(img_concatenate.shape[0] / 4))
    cv2.imshow('Stereo', img_concatenate)
    cv2.waitKey(0)
    cv2.destroyWindow('Stereo')


def save_array_to_csv(array, filename):
    """
    Save a 2D NumPy array to a CSV file.

    :param array: 2D numpy array
    :param filename: Output CSV filename
    """
    # Save the 2D array as a CSV file
    np.savetxt(filename, array, delimiter=',')
    print(f"Array saved to {filename}")


def load_array_from_csv(filename, delimiter=','):
    """
    Load a 2D NumPy array from a CSV file.

    :param filename: Input CSV filename
    :return: 2D numpy array
    """
    # Load the array from the CSV file
    array = np.loadtxt(filename, delimiter=delimiter)
    return array


def plot_3d_image(image):
    """
    Plot image on 3D graph where Z is the intensity of pixel
    Parameters:
        image: image to be plotted
    """
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


def plot_images(left_image, right_image, uv_points_l, uv_points_r):
    left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
    right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 12))
    for uv in range(uv_points_l.shape[1]):
        ax1.imshow(left_image)
        circle = plt.Circle((uv_points_l[0, uv], uv_points_l[1, uv]), 5, color='r', fill=False, lw=2)
        ax1.add_patch(circle)  # Add circle to the plot
        ax1.axis('off')
    for uv in range(uv_points_r.shape[1]):
        ax2.imshow(right_image)
        circle = plt.Circle((uv_points_r[0, uv], uv_points_r[1, uv]), 5, color='r', fill=False, lw=2)
        ax2.add_patch(circle)  # Add circle to the plot
        ax2.axis('off')

    plt.tight_layout()
    plt.show()


def plot_zscan_phi(phi_map):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9))
    plt.title('Z step for abs(phase_left - phase_right)')
    for j in range(len(phi_map)):
        if j < len(phi_map) // 2:
            ax1.plot(phi_map[j], label="{}".format(j))
            circle = plt.Circle((np.argmin(phi_map[j]), np.min(phi_map[j])), 0.05, color='r', fill=False, lw=2)
            ax1.add_patch(circle)  # Add circle to the plot
            ax1.set_ylabel('correlation [%]')
            ax1.set_xlabel('z steps')
            ax1.grid(True)
            # ax1.legend()
        if j >= len(phi_map) // 2:
            ax2.plot(phi_map[j], label="{}".format(j))
            circle = plt.Circle((np.argmin(phi_map[j]), np.min(phi_map[j])), 0.05, color='r', fill=False, lw=2)
            ax2.add_patch(circle)  # Add circle to the plot
            ax2.set_xlabel('z steps')
            ax2.set_ylabel('correlation [%]')
            ax2.grid(True)
            # ax2.legend()
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()


def plot_zscan_correl(correl_ar, list_of_points=None, nimgs=10, title='Title'):

    z_size = 1000

    # Create a figure with 2 subplots (1 row, 2 columns)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9))
    plt.title('{} \n {} images'.format(title, nimgs))
    # First graph
    for k in range(correl_ar.shape[0] // z_size):
        if k < correl_ar.shape[0] // (2 * z_size):
            ax1.plot(correl_ar[k * z_size:(k + 1) * z_size], label="{}".format(k))
            ax1.set_xlabel('z steps')
            ax1.set_ylabel('correlation [%]')
            circle = plt.Circle((np.argmin(correl_ar[k]), np.min(correl_ar[k])), 0.05, color='r', fill=False, lw=2)
            ax1.add_patch(circle)  # Add circle to the plot
            ax1.grid(True)
            # ax1.legend()
        if k >= correl_ar.shape[0] // (2 * z_size):
            ax2.plot(correl_ar[k * z_size:(k + 1) * z_size], label="{}".format(k))
            ax2.set_xlabel('z steps')
            ax2.set_ylabel('correlation [%]')
            circle = plt.Circle((np.argmin(correl_ar[k]), np.min(correl_ar[k])), 0.05, color='r', fill=False, lw=2)
            ax2.add_patch(circle)  # Add circle to the plot
            ax2.grid(True)
            # ax2.legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()


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


def plot_2d_planes(xyz):
    # Extract X, Y, Z coordinates
    # Extract X, Y, Z coordinates
    X = xyz[:, 0]
    Y = xyz[:, 1]
    Z = xyz[:, 2]

    # Create a figure with 2 rows and 3 columns for subplots
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))

    # Plot XY projection
    ax[0, 0].scatter(X, Y, c='b', marker='o')
    ax[0, 0].set_xlabel('X')
    ax[0, 0].set_ylabel('Y')
    ax[0, 0].set_title('XY Projection')
    ax[0, 0].grid()

    # Plot XZ projection
    ax[0, 1].scatter(X, Z, c='r', marker='o')
    ax[0, 1].set_xlabel('X')
    ax[0, 1].set_ylabel('Z')
    ax[0, 1].set_title('XZ Projection')
    ax[0, 1].grid()

    # Plot YZ projection
    ax[0, 2].scatter(Y, Z, c='g', marker='o')
    ax[0, 2].set_xlabel('Y')
    ax[0, 2].set_ylabel('Z')
    ax[0, 2].set_title('YZ Projection')
    ax[0, 2].grid()

    # Plot X distribution
    ax[1, 0].hist(X, bins=20, color='b', alpha=0.7)
    ax[1, 0].set_xlabel('X')
    ax[1, 0].set_ylabel('Frequency')
    ax[1, 0].set_title('X Distribution')

    # Plot Y distribution
    ax[1, 1].hist(Y, bins=20, color='r', alpha=0.7)
    ax[1, 1].set_xlabel('Y')
    ax[1, 1].set_ylabel('Frequency')
    ax[1, 1].set_title('Y Distribution')

    # Plot Z distribution
    ax[1, 2].hist(Z, bins=20, color='g', alpha=0.7)
    ax[1, 2].set_xlabel('Z')
    ax[1, 2].set_ylabel('Frequency')
    ax[1, 2].set_title('Z Distribution')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()


def plot_hist(left, right):
    if left.shape.__len__() != right.shape.__len__():
        print("Images are not colored")
        return False
    if left.shape.__len__() > 2:
        print("Images are colored")

        hist_r = []
        hist_l = []
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

        for n in range(left.shape[2]):
            hist_r.append(cv2.calcHist([left[:, :, n]], [0], None, [256], [0, 256]))
            hist_l.append(cv2.calcHist([right[:, :, n]], [0], None, [256], [0, 256]))

        # Plot on the first subplot
        ax1.plot(hist_l[0], label='blue', color='blue')
        ax1.plot(hist_l[1], label='green', color='green')
        ax1.plot(hist_l[2], label='red', color='red')
        ax1.set_title('Left histogram')
        ax1.set_ylabel('N° of pixels')
        ax1.legend()
        ax1.grid(True)

        # Plot on the second subplot
        ax2.plot(hist_r[0], label='blue', color='blue')
        ax2.plot(hist_r[1], label='green', color='green')
        ax2.plot(hist_r[2], label='red', color='red')
        ax2.set_title('Right histogram')
        ax2.set_ylabel('N° of pixels')
        ax2.legend()
        ax2.grid(True)

    elif left.shape.__len__() < 3:

        hist_r = (cv2.calcHist([left], [0], None, [256], [0, 256]))
        hist_l = (cv2.calcHist([right], [0], None, [256], [0, 256]))

        plt.plot(hist_r, label='right', color='blue')
        plt.plot(hist_l, label='left', color='green')
        plt.legend()
        plt.grid(True)
    plt.show()


def crop_img2proj_points(image, uv_points):
    u = (int(np.min(uv_points[0, :])), int(np.max(uv_points[0, :])))
    v = (int(np.min(uv_points[1, :])), int(np.max(uv_points[1, :])))
    croped_img = image[v[0]:v[1], u[0]:u[1]]
    return croped_img


def plot_point_correl(xyz, ho):
    z_size = np.unique(xyz[:, 2]).shape[0]
    n_x = np.unique(xyz[:, 0]).shape[0]
    for x_val in range(n_x):
        plt.figure()
        plt.plot(ho[x_val * z_size:z_size * (x_val + 1)])
        plt.grid(True)
        plt.title(xyz[x_val * z_size, :])
    plt.show()


def plot_points_on_image(image, points, color=(0, 255, 0), radius=5, thickness=1):
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
    for (u, v) in points.T:
        # Draw a circle for each point on the image
        cv2.circle(output_image, (int(u), int(v)), radius, color, thickness)
    return output_image


def show_stereo_images(left, right, name='Rectified Images'):
    combined_image = np.concatenate((left, right), axis=1)
    # combined_image = cv2.line(combined_image, (0,1460), (8000, 1460), (0, 255, 0))
    # combined_image = cv2.line(combined_image, (0,1431), (8000, 1431), (0, 255, 0))
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, int(combined_image.shape[1] / 4), int(combined_image.shape[0] / 4))
    cv2.imshow(name, combined_image)
    cv2.waitKey(0)


def mask_images(left_image, right_image, thres=180):
    if (left_image.shape.__len__() == right_image.shape.__len__()) and right_image.shape.__len__() < 3:
        mask_left = cv2.threshold(cv2.GaussianBlur(left_image, (3, 3), 0), thres, 255, cv2.THRESH_BINARY)[1]
        mask_right = cv2.threshold(cv2.GaussianBlur(right_image, (3, 3), 0), thres, 255, cv2.THRESH_BINARY)[1]
    else:
        # Get only the red spectrum to mask
        mask_left = cv2.threshold(cv2.GaussianBlur(left_image[:, :, 2], (3, 3), 0), thres, 255, cv2.THRESH_BINARY)[1]
        mask_right = cv2.threshold(cv2.GaussianBlur(right_image[:, :, 2], (3, 3), 0), thres, 255, cv2.THRESH_BINARY)[1]

    mask_left = cv2.erode(mask_left, (3, 3), iterations=2)
    mask_right = cv2.erode(mask_right, (3, 3), iterations=2)
    mask_left = cv2.morphologyEx(mask_left, cv2.MORPH_CLOSE, kernel=np.ndarray([5, 5], np.uint8))
    mask_right = cv2.morphologyEx(mask_right, cv2.MORPH_CLOSE, kernel=np.ndarray([5, 5], np.uint8))

    return mask_left, mask_right

# def main():
#     # for i in range(3):
#     #     for j in range(3):
#             # print(i, j)
#         # file = '/home/daniel/reshaped_2pts_{}_{}.txt'.format(i, j)
#         file = '/home/daniel/reshaped_1pts.txt'
#         # if not os.path.exists(file):
#         #     print('File not found')
#         #     continue
#         i=j=0
#         # Load the data from the CSV file
#         data = load_array_from_csv(file, delimiter=',')
#         print('Data readed from file, size: ', data.shape)
        

#         # Filter data_arr to only include arrays that contain values above 0.5
#         data_arr = np.array_split(data, data.shape[0] // 1000)
#         filtered_data_arr = [arr for arr in data_arr if np.any(arr > 0.5)]
#         filtered = np.hstack(filtered_data_arr)
#         print('Filtered data size: ', filtered.shape)

#         plot_zscan_correl(data, title='All data{},{}'.format(i, j))
#         plot_zscan_correl(filtered, title='Filtered data {},{}'.format(i, j))
#         plt.show()




# if __name__ == '__main__':
#     main()
