import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Load images ---
left_img = cv2.imread("/home/daniel/Pictures/correl/left/L01.png", cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread("/home/daniel/Pictures/correl/right/R01.png", cv2.IMREAD_GRAYSCALE)

if left_img is None or right_img is None:
    raise FileNotFoundError("Could not load 'left.png' or 'right.png'")

# --- Create window and trackbars ---
cv2.namedWindow("CLAHE Viewer", cv2.WINDOW_NORMAL)
cv2.namedWindow("Original Images", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Original Images", 800, 400)

def nothing(x):
    pass

cv2.createTrackbar("ClipLimit", "CLAHE Viewer", 10, 50, nothing)  # Range 0.0 - 10.0
cv2.createTrackbar("TileGridSize", "CLAHE Viewer", 8, 32, nothing)  # Range 1 - 32

def plot_histograms(left, right, left_eq, right_eq, title):
    """
    Plot histograms for the original and equalized images.
    """
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(left.ravel(), bins=256, range=(0, 256), color='blue', alpha=0.7, label='left')
    plt.hist(right.ravel(), bins=256, range=(0, 256), color='green', alpha=0.7, label='right')
    plt.title(f"{title} - Original Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(left_eq.ravel(), bins=256, range=(0, 256), color='blue', alpha=0.7, label='left')
    plt.hist(right_eq.ravel(), bins=256, range=(0, 256), color='green', alpha=0.7, label='Right')
    plt.title(f"{title} - Equalized Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend()

    plt.tight_layout()
    plt.show()

while True:
    # --- Read trackbar values ---
    clip_limit = cv2.getTrackbarPos("ClipLimit", "CLAHE Viewer")
    tile_grid_size = cv2.getTrackbarPos("TileGridSize", "CLAHE Viewer")
    tile_grid_size = max(1, tile_grid_size)  # Ensure >= 1

    # --- Apply CLAHE ---
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    left_eq = clahe.apply(left_img)
    right_eq = clahe.apply(right_img)

    # --- Stack images side-by-side ---
    combined = np.hstack((left_eq, right_eq))
    combo_original = np.hstack((left_img, right_img))

    # --- Display ---
    cv2.imshow("CLAHE Viewer", combined)
    cv2.imshow("Original Images", combo_original)

    # --- Plot histograms ---
    key = cv2.waitKey(30)
    if key == ord('h'):  # Press 'h' to show histograms
        plot_histograms(left_img, right_img, left_eq, right_eq, "Image")

    if key == 27:  # ESC key
        break

cv2.destroyAllWindows()
