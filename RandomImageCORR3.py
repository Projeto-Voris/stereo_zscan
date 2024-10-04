import numpy as np
import random
import matplotlib.pyplot as plt
from noise import pnoise2
from scipy.stats import rankdata, norm, spearmanr
import cv2
import os


class NoiseImageGenerator:
    def __init__(self, image_width, image_height, frequency):
        self.image_width = image_width
        self.image_height = image_height
        self.frequency = frequency

    def generate_noise_image(self, debug=False):
        # Prepare grid coordinates
        i, j = np.meshgrid(np.arange(self.image_height), np.arange(self.image_width), indexing='ij')

        # Generate a new random base seed
        random_base = random.randint(30, 50)

        # Vectorized function to compute noise for each coordinate
        noise_function = np.vectorize(lambda x, y: pnoise2(x / self.frequency, y / self.frequency, octaves=8,
                                                           persistence=0.4, lacunarity=6.0,
                                                           repeatx=self.image_width, repeaty=self.image_height,
                                                           base=random_base))

        # Apply noise function to the entire grid
        noise_values = noise_function(i, j)

        # Normalize and convert to uint8
        pattern_image = np.clip((noise_values + 0.35) * 255, 0, 255).astype(np.uint8)
        if debug:
            cv2.imshow('noise', pattern_image)
            cv2.waitKey(0)
        cv2.destroyAllWindows()
        return pattern_image

    def generate_stacked_images(self, num_images):
        # Create a 3D matrix to store multiple images
        stacked_images = np.zeros((num_images, self.image_height, self.image_width), dtype=np.uint8)

        # Generate each image and store it in the 3D matrix
        for idx in range(num_images):
            stacked_images[idx, :, :] = self.generate_noise_image(debug=True)

        return stacked_images

    def read_images(self, path, images_list, n_images, visualize=False):
        """
        Read all images from the specified path and stack them into a single array.
        Parameters:
            path: (string) path to images folder.
            images_list: (list of strings) list of image names.
        Returns:
            images: (height, width, number of images) array of images.
        """
        # Read all images using list comprehension
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        images = [clahe.apply(cv2.imread(os.path.join(path, str(img_name)), cv2.IMREAD_GRAYSCALE)) for img_name in
                  images_list[0:n_images]]

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

    def calculate_pearson_correlation_row(self, stacked_images, row, pixel_position):
        """
        Calculate Pearson correlation between the vector at pixel_position and
        all other vectors on the same row across the stacked images.
        """
        reference_vector = stacked_images[pixel_position, row, :]
        correlations = []

        for col in range(stacked_images.shape[0]):  # Loop over all columns in the row
            current_vector = stacked_images[col, row, :]
            correlation = np.corrcoef(reference_vector, current_vector)[0, 1]
            correlations.append(correlation)

        return correlations

    def calculate_pearson_stero(self, left_images, right_images, pixel_position=(0, 0)):
        row, col = pixel_position
        correlations = []
        max = []
        for col in range(left_images.shape[0]):
            correlation = np.asarray(
                [np.corrcoef(left_images[col, i, :], right_images[col, i, :])[0, 1] for i in
                 range(left_images.shape[1])],
                np.float32)
            correlations.append(correlation)
        for correl in correlations:
            max.append(np.nanmax(correl))
        return correlations[np.argmax(max)], (np.argmax(correlations[np.argmax(max)]), np.argmax(max))

    def calculate_xicor_correlation_row(self, stacked_images, row, pixel_position):
        """
        Calculate XICOR correlation between the vector at pixel_position and
        all other vectors on the same row across the stacked images.
        """
        reference_vector = stacked_images[pixel_position, row, :]
        correlations = []

        for col in range(stacked_images.shape[0]):  # Loop over all columns in the row
            current_vector = stacked_images[col, row, :]
            correlation, _ = xicor(reference_vector, current_vector)
            correlations.append(correlation)

        return correlations

    def calculate_spearman_correlation_row(self, stacked_images, row, col):
        """
        Calculate Spearman correlation between the vector at pixel_position and
        all other vectors on the same row across the stacked images.
        """
        reference_vector = stacked_images[col, row, :]
        correlations = []

        for coll in range(stacked_images.shape[2]):  # Loop over all columns in the row
            current_vector = stacked_images[coll, row, :]
            correlation, _ = spearmanr(reference_vector, current_vector)
            correlations.append(correlation)

        return correlations

    def plot_correlation_graphs(self, row, pixel_position, num_images_list):
        """
        Plot Pearson, XICOR, and Spearman correlation graphs for different numbers of stacked images.

        Parameters:
        row (int): The row in the image to calculate correlations for.
        pixel_position (int): The column position of the reference vector.
        num_images_list (list): A list of different num_images values for which to generate graphs.
        """

        # Plot Pearson correlation graph
        plt.figure(figsize=(10, 6))
        for num_images in num_images_list:
            # stacked_images = self.generate_stacked_images(num_images)
            pearson_correlations = self.calculate_pearson_correlation_row(stacked_images, row, pixel_position)
            plt.plot(pearson_correlations, label=f'Pearson - {num_images} images')

        plt.title(f'Pearson Correlation Across Row {row} for Different Numbers of Stacked Images')
        plt.xlabel('Pixel Position (Column)')
        plt.ylabel('Correlation Coefficient')
        plt.grid(True)
        plt.legend()
        plt.show()

        # Plot XICOR correlation graph
        plt.figure(figsize=(10, 6))
        for num_images in num_images_list:
            stacked_images = self.generate_stacked_images(num_images)
            xicor_correlations = self.calculate_xicor_correlation_row(stacked_images, row, pixel_position)
            plt.plot(xicor_correlations, label=f'XICOR - {num_images} images')

        plt.title(f'XICOR Correlation Across Row {row} for Different Numbers of Stacked Images')
        plt.xlabel('Pixel Position (Column)')
        plt.ylabel('Correlation Coefficient')
        plt.grid(True)
        plt.legend()
        plt.show()

        # Plot Spearman correlation graph
        plt.figure(figsize=(10, 6))
        for num_images in num_images_list:
            stacked_images = self.generate_stacked_images(num_images)
            spearman_correlations = self.calculate_spearman_correlation_row(stacked_images, row, pixel_position)
            plt.plot(spearman_correlations, label=f'Spearman - {num_images} images')

        plt.title(f'Spearman Correlation Across Row {row} for Different Numbers of Stacked Images')
        plt.xlabel('Pixel Position (Column)')
        plt.ylabel('Correlation Coefficient')
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_correlation_graphs_realimg(self, row, pixel_position, image_list_number, path, image_list):
        """
        Plot Pearson, XICOR, and Spearman correlation graphs for different numbers of stacked images.

        Parameters:
        row (int): The row in the image to calculate correlations for.
        pixel_position (int): The column position of the reference vector.
        num_images_list (list): A list of different num_images values for which to generate graphs.
        """
        for num_images in image_list_number:
            stacked_images = self.read_images(path=path, n_images=num_images, images_list=image_list)
            # Plot Pearson correlation graph
            plt.figure(figsize=(10, 6))
            pearson_correlations = self.calculate_pearson_correlation_row(stacked_images, row, pixel_position)
            plt.plot(pearson_correlations, label=f'Pearson - {num_images} images')

            plt.title(f'Pearson Correlation Across Row {row} for Different Numbers of Stacked Images')
            plt.xlabel('Pixel Position (Column)')
            plt.ylabel('Correlation Coefficient')
            plt.grid(True)
            plt.legend()
            plt.show()

            # Plot XICOR correlation graph
            plt.figure(figsize=(10, 6))
            xicor_correlations = self.calculate_xicor_correlation_row(stacked_images, row, pixel_position)
            plt.plot(xicor_correlations, label=f'XICOR - {num_images} images')

            plt.title(f'XICOR Correlation Across Row {row} for Different Numbers of Stacked Images')
            plt.xlabel('Pixel Position (Column)')
            plt.ylabel('Correlation Coefficient')
            plt.grid(True)
            plt.legend()
            plt.show()

    def plot_correl_stereo_img(self, row, pixel_position, image_list_number, path, left_list, right_list):
        """
        Plot Pearson, XICOR, and Spearman correlation graphs for different numbers of stacked images.

        Parameters:
        row (int): The row in the image to calculate correlations for.
        pixel_position (int): The column position of the reference vector.
        num_images_list (list): A list of different num_images values for which to generate graphs.
        """
        plt.figure(figsize=(10, 6))
        for num_images in image_list_number:
            left_images = self.read_images(path=os.path.join(path, 'left'), n_images=num_images, images_list=left_list)
            right_images = self.read_images(path=os.path.join(path, 'right'), n_images=num_images,
                                            images_list=right_list)
            # Plot Pearson correlation graph
            pearson_correlations, point = self.calculate_pearson_stero(left_images=left_images,
                                                                       right_images=right_images,
                                                                       pixel_position=(row, pixel_position))
            plt.plot(pearson_correlations, label=f'Pearson - {num_images} images')

            plt.title(f'Pearson Correlation Stereo for Different Numbers of Stacked Images')
            plt.xlabel('Pixel Position (Column)')
            plt.ylabel('Correlation Coefficient')
            plt.grid(True)
            plt.legend()
            plt.show()
            print('Image number {} to correl point {}, {}'.format(num_images, point[0], point[1]))

            # self.plot_stereo_correl_pt(left=left_images[:, :, 0], right=right_images[:, :, 0],
            #                            pt_l=(row, pixel_position), pt_r=point)

    def plot_stereo_correl_pt(self, left, right, pt_l, pt_r):
        cv2.circle(left, (pt_l[0], pt_l[1]), 5, (255, 0, 255), -1)
        cv2.circle(right, (pt_r[0], pt_r[1]), 5, (255, 0, 255), -1)
        concat = np.hstack((left, right))
        plt.figure(figsize=(10, 6))
        plt.imshow(concat, cmap='gray')
        plt.show()

# Updated XICOR correlation function
def xicor(x, y, ties="auto"):
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    n = len(y)

    if len(x) != n:
        raise IndexError(
            f"x, y length mismatch: {len(x)}, {len(y)}"
        )

    if ties == "auto":
        ties = len(np.unique(y)) < n
    elif not isinstance(ties, bool):
        raise ValueError(
            f"expected ties either \"auto\" or boolean, "
            f"got {ties} ({type(ties)}) instead"
        )

    y = y[np.argsort(x)]
    r = rankdata(y, method="ordinal")
    nominator = np.sum(np.abs(np.diff(r)))

    if ties:
        l = rankdata(y, method="max")
        denominator = 2 * np.sum(l * (n - l))
        nominator *= n
    else:
        denominator = np.power(n, 2) - 1
        nominator *= 3

    statistic = 1 - nominator / denominator  # upper bound is (n - 2) / (n + 1)
    p_value = norm.sf(statistic, scale=2 / 5 / np.sqrt(n))

    return statistic, p_value


def main():
    # Usage Example
    image_generator = NoiseImageGenerator(image_width=500, image_height=500, frequency=7)

    # Define the row and pixel position to use for correlation calculation
    row = 100  # Example row
    pixel_position = 250  # Example reference pixel in the same row

    # Define different numbers of stacked images to plot
    num_images_list = [5, 15, 30]

    images_path = '/home/daniel/Insync/daniel.regner@labmetro.ufsc.br/Google Drive - Shared drives/VORIS  - Equipe/Sistema de Medição 3 - Stereo Ativo - Projeção Laser/Imagens/Testes/SM3-20240919 - noise 4.0'
    Nimg = [20, 50, 100]
    DEBUG = False
    # Identify all images from path file
    left_images = sorted(os.listdir(os.path.join(images_path, 'left')))

    right_images = sorted(os.listdir(os.path.join(images_path, 'right')))

    # image_generator.plot_correlation_graphs_realimg(row=row, pixel_position=pixel_position,
    #                                                 image_list_number=Nimg, path=os.path.join(images_path,'left'), image_list=left_images)

    image_generator.plot_correl_stereo_img(row=row, pixel_position=pixel_position,
                                           image_list_number=Nimg, path=images_path, left_list=left_images,
                                           right_list=right_images)

    # Plot the correlation graphs for Pearson, XICOR, and Spearman correlations for different numbers of images
    # image_generator.plot_correlation_graphs(row, pixel_position, num_images_list)


if __name__ == '__main__':
    main()
