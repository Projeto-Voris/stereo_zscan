import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


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

    mask_left = cv2.erode(mask_left, (3,3), iterations=2)
    mask_right = cv2.erode(mask_right, (3,3), iterations=2)
    mask_left = cv2.morphologyEx(mask_left, cv2.MORPH_CLOSE, kernel=np.ndarray([5,5], np.uint8))
    mask_right = cv2.morphologyEx(mask_right, cv2.MORPH_CLOSE, kernel=np.ndarray([5,5], np.uint8))


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
