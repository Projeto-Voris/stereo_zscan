import cv2
import numpy as np
import yaml


def add_grid_lines(image, interval=50, color=(0, 255, 0), thickness=1):
    """
    Add grid lines to an image at specified intervals.

    :param image: Input image (numpy array).
    :param interval: Interval between lines in pixels.
    :param color: Color of the lines in BGR format (default is green).
    :param thickness: Thickness of the lines.
    :return: Image with grid lines added.
    """
    # Make a copy of the image to avoid modifying the original
    img_with_lines = image.copy()

    height, width = img_with_lines.shape[:2]

    # Draw horizontal lines
    for y in range(0, height, interval):
        cv2.line(img_with_lines, (0, y), (width, y), color, thickness)

    # # Draw vertical lines
    # for x in range(0, width, interval):
    #     cv2.line(img_with_lines, (x, 0), (x, height), color, thickness)

    return img_with_lines


def draw_keypoints(left, right):
    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(left, None)
    kp2, des2 = sift.detectAndCompute(right, None)
    # Match keypoints in both images
    # Based on: https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]
    # Draw the keypoint matches between both pictures
    # Still based on: https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=cv2.DrawMatchesFlags_DEFAULT)

    img3 = cv2.drawMatchesKnn(left, kp1, right, kp2, matches, None, **draw_params)

    # plt.imshow(img3,),plt.show()
    cv2.namedWindow("Keypoint matches", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Keypoint matches", 1920, 1080)
    cv2.imshow("Keypoint matches", img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def load_camera_params(yaml_file):
    # Load the YAML file
    with open(yaml_file) as file:  # Replace with your file path
        params = yaml.safe_load(file)

        # Parse the matrices
    Kl = np.array(params['camera_matrix_left'], dtype=np.float64)
    Dl = np.array(params['dist_coeffs_left'], dtype=np.float64)
    Pl = np.array(params['proj_matrix_left'], dtype=np.float64)

    Kr = np.array(params['camera_matrix_right'], dtype=np.float64)
    Dr = np.array(params['dist_coeffs_right'], dtype=np.float64)
    Pr = np.array(params['proj_matrix_right'], dtype=np.float64)

    R = np.array(params['R'], dtype=np.float64)
    T = np.array(params['T'], dtype=np.float64)

    return Kl, Kr, Dl, Dr, Pl, Pr, R, T


def main():
    left_image = cv2.imread('images/20240809/left/L003.png', 0)
    right_image = cv2.imread('images/20240809/right/R003.png',0)
    cv2.equalizeHist(left_image, left_image)
    cv2.equalizeHist(right_image, right_image)

    K1, K2, dist1, dist2, P1, P2, R, T = load_camera_params(yaml_file='cfg/20240809.yaml')

    # Perform stereo rectification
    Rr1, Rr2, Pr1, Pr2, Q, ROI1, ROI2 = cv2.stereoRectify(K1, dist1, K2, dist2,
                                                          (left_image.shape[1], left_image.shape[0]), R, T, alpha=1)

    # Compute the undistortion and rectification transformation map
    map1x, map1y = cv2.initUndistortRectifyMap(K1, dist1, Rr1, Pr1, (left_image.shape[1], left_image.shape[0]),
                                               cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, dist2, Rr2, Pr2, (left_image.shape[1], left_image.shape[0]),
                                               cv2.CV_32FC1)

    # Apply the rectification maps to the images
    rectified_left = cv2.remap(left_image, map1x, map1y, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(right_image, map2x, map2y, cv2.INTER_LINEAR)

    # Display rectified images side by side
    combined_image = np.concatenate((rectified_left, rectified_right), axis=1)
    combined_image = add_grid_lines(combined_image, interval=300)
    cv2.namedWindow('Rectified Images', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Rectified Images', int(combined_image.shape[1]/4), int(combined_image.shape[0]/4))
    cv2.imshow('Rectified Images', combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    draw_keypoints(rectified_left, rectified_right)


if __name__ == '__main__':
    main()
