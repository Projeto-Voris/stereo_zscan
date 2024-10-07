from sys import path_hooks

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def save_array_to_csv(array, filename):
    """
    Save a 2D NumPy array to a CSV file.

    :param array: 2D numpy array
    :param filename: Output CSV filename
    """
    # Save the 2D array as a CSV file
    np.savetxt(filename, array, delimiter=',')
    print(f"Array saved to {filename}")

def load_array_from_csv(filename):
    """
    Load a 2D NumPy array from a CSV file.

    :param filename: Input CSV filename
    :return: 2D numpy array
    """
    # Load the array from the CSV file
    array = np.loadtxt(filename, delimiter=',')
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


def plot_images(left_image, right_image):
    left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
    right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16, 12))

    ax1.imshow(left_image)
    ax1.axis('off')
    ax2.imshow(right_image)
    ax2.axis('off')

    plt.tight_layout()
    plt.show()


def plot_zscan_phi(phi_map):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9))
    plt.title('Z step for abs(phase_left - phase_right)')
    for j in range(len(phi_map)):
        if j < len(phi_map) // 2:
            ax1.plot(phi_map[j], label="{}".format(j))
            ax1.set_xlabel('z steps')
            ax1.set_ylabel('correlation [%]')
            ax1.grid(True)
            ax1.legend()
        if j >= len(phi_map) // 2:
            ax2.plot(phi_map[j], label="{}".format(j))
            ax2.set_xlabel('z steps')
            ax2.set_ylabel('correlation [%]')
            ax2.grid(True)
            ax2.legend()
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

def plot_zscan(correl_ar, xyz_points, list_of_points=None):
    z_size = np.unique(xyz_points[:, 2]).shape[0]
    if list_of_points is None:
        list_of_points = np.arange(np.unique(xyz_points[:, 0]).shape[0])

        # Create a figure with 2 subplots (1 row, 2 columns)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9))

        # First graph
        for k in range(xyz_points.shape[0] // z_size):
            if k < xyz_points.shape[0] // (2 * z_size):
                ax1.plot(correl_ar[k * z_size:(k + 1) * z_size], label="{}".format(k))
                ax1.set_xlabel('z steps')
                ax1.set_ylabel('correlation [%]')
                ax1.grid(True)
                ax1.legend()
            if k >= xyz_points.shape[0] // (2 * z_size):
                ax2.plot(correl_ar[k * z_size:(k + 1) * z_size], label="{}".format(k))
                ax2.set_xlabel('z steps')
                ax2.set_ylabel('correlation [%]')
                ax2.grid(True)
                ax2.legend()

        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.show()


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
    for (u, v, _) in points.T:
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


def main():
    path = 'images/SM3-20240815_1'
    left_images = os.listdir(os.path.join(path, 'left'))
    right_images = os.listdir(os.path.join(path, 'right'))

    left_image = cv2.imread(os.path.join(path, 'left', left_images[0]), 0)
    right_image = cv2.imread(os.path.join(path, 'right', right_images[0]), 0)

    # plot_hist(left_image, right_image)

    masked_l, masked_r = mask_images(left_image, right_image, thres=170)

    show_stereo_images(left_image, right_image, name='original')
    show_stereo_images(masked_l, masked_r, name='masked')

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
