import numpy as np
import random
import matplotlib.pyplot as plt
from noise import pnoise2
from scipy.stats import rankdata, norm, spearmanr
import cv2

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

    def calculate_pearson_correlation_row(self, stacked_images, row, pixel_position):
        """
        Calculate Pearson correlation between the vector at pixel_position and
        all other vectors on the same row across the stacked images.
        """
        reference_vector = stacked_images[:, row, pixel_position]
        correlations = []

        for col in range(stacked_images.shape[2]):  # Loop over all columns in the row
            current_vector = stacked_images[:, row, col]
            correlation = np.corrcoef(reference_vector, current_vector)[0, 1]
            correlations.append(correlation)

        return correlations

    def calculate_xicor_correlation_row(self, stacked_images, row, pixel_position):
        """
        Calculate XICOR correlation between the vector at pixel_position and
        all other vectors on the same row across the stacked images.
        """
        reference_vector = stacked_images[:, row, pixel_position]
        correlations = []

        for col in range(stacked_images.shape[2]):  # Loop over all columns in the row
            current_vector = stacked_images[:, row, col]
            correlation, _ = xicor(reference_vector, current_vector)
            correlations.append(correlation)

        return correlations

    def calculate_spearman_correlation_row(self, stacked_images, row, pixel_position):
        """
        Calculate Spearman correlation between the vector at pixel_position and
        all other vectors on the same row across the stacked images.
        """
        reference_vector = stacked_images[:, row, pixel_position]
        correlations = []

        for col in range(stacked_images.shape[2]):  # Loop over all columns in the row
            current_vector = stacked_images[:, row, col]
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
            stacked_images = self.generate_stacked_images(num_images)
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

# Usage Example
image_generator = NoiseImageGenerator(image_width=500, image_height=500, frequency=7)

# Define the row and pixel position to use for correlation calculation
row = 100  # Example row
pixel_position = 250  # Example reference pixel in the same row

# Define different numbers of stacked images to plot
num_images_list = [5, 15, 30]

# Plot the correlation graphs for Pearson, XICOR, and Spearman correlations for different numbers of images
image_generator.plot_correlation_graphs(row, pixel_position, num_images_list)
