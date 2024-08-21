import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import imutils
# Scripts to debugg and rectify images
import debugger
import rectify_matrix


def detect_orb_dots(image_left, image_right):
    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Detect ORB keypoints and descriptors in both images
    keypoints_left, descriptors_left = orb.detectAndCompute(image_left, None)
    keypoints_right, descriptors_right = orb.detectAndCompute(image_right, None)

    # Create a BFMatcher object to match descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors between the two images
    matches = bf.match(descriptors_left, descriptors_right)

    # Sort the matches based on distance (the lower, the better)
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw the first 10 matches for visualization
    matched_image = cv2.drawMatches(image_left, keypoints_left, image_right, keypoints_right, matches[:50], None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return matched_image


def detect_dots(image):
    """
    Detect the dots in the image using blob detection based on Area
    """
    # Convert to grayscale if the image is not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Set up the SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector.Params()
    params.filterByArea = True
    params.minArea = 1
    params.maxArea = 1500

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector.create(params)

    # Detect blobs
    keypoints = detector.detect(gray)

    # Extract points
    points = np.array([kp.pt for kp in keypoints], dtype=np.uint16)

    return points


def detect_dots_findcontours(image):
    contours, herarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    points = []
    # loop over the contours
    for c in contours:
        # compute the center of the contour
        M = cv2.moments(c)
        if M["m00"] != 0:
            points.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))
        else:
            continue
    return points, contours


def search_window(image_left, image_right, point, window_size=5, search_range=1000):
    # Convert point coordinates to integers
    x, y = int(point[0]), int(point[1])
    half_window = window_size // 2

    # Define the window in the left image
    left_window = image_left[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1]

    # Search for the best matching window in the right image
    best_match = None
    best_score = float('inf')

    for dx in range(-search_range, search_range + 1):
        x_right = x + dx
        if x_right - half_window < 0 or x_right + half_window >= image_right.shape[1]:
            continue
        right_window = image_right[y - half_window:y + half_window + 1, x_right - half_window:x_right + half_window + 1]

        if right_window.shape == left_window.shape:
            ssd = np.sum((left_window - right_window) ** 2)
            if ssd < best_score:
                best_score = ssd
                best_match = (x_right, y)

    return best_match


def match_points(points_left, points_right, threshold):
    """
    Match points between two images based on distance.
    """
    matched_left = []
    matched_right = []

    for pl in points_left:
        distances = np.linalg.norm(points_right - pl, axis=1)
        min_index = np.argmin(distances)

        if distances[min_index] < threshold:
            matched_left.append(pl)
            matched_right.append(points_right[min_index])

    return np.array(matched_left), np.array(matched_right)

def match_points2(points_left, points_right, threshold):
    matched_left = []
    matched_right = []
    for pL in points_left:
        for pR in points_right:
            if abs(pL[1] -pR[1]) < 5:
                distances = abs((pL[0]-1397)-(pR[0]-23))
                if distances < threshold:
                    matched_left.append([pL[0], pL[1]])
                    matched_right.append([pR[0], pR[1]])

    seen = set()
    m_left = []
    for item in matched_left:
        t = tuple(item)
        if t not in seen:
            m_left.append(item)
            seen.add(t)
    m_right = []
    seen = set()
    for item in matched_right:
        t = tuple(item)
        if t not in seen:
            m_right.append(item)
            seen.add(t)
    # *m_left, = map(list, {*map(tuple, matched_left)})
    # *m_right, = map(list, {*map(tuple, matched_right)})
    return np.array(m_left), np.array(m_right)
def draw_matched_points(image_left, image_right, m_p_left, m_p_right, color=(0, 255, 0), radius=5, thickness=2):
    # Convert the images to color if they are grayscale
    if len(image_left.shape) == 2:
        image_left = cv2.cvtColor(image_left, cv2.COLOR_GRAY2BGR)
    if len(image_right.shape) == 2:
        image_right = cv2.cvtColor(image_right, cv2.COLOR_GRAY2BGR)

    # Draw the matched points on both images
    for pt_left, pt_right in zip(m_p_left, m_p_right):
        # Draw circles on the left image
        cv2.circle(image_left, tuple(int(x) for x in pt_left), radius, color, thickness)
        # Draw circles on the right image
        cv2.circle(image_right, tuple(int(x) for x in pt_right), radius, (0,255,0), thickness)
    return image_left, image_right


def main():
    path = '/home/daniel/PycharmProjects/stereo_active/images/SM3-20240815_1'
    yaml_file = 'cfg/20240815_rect_1.yaml'
    left_images = sorted(os.listdir(os.path.join(path, 'left')))
    right_images = sorted(os.listdir(os.path.join(path, 'right')))

    left_image = cv2.imread(os.path.join(path, 'left', left_images[0]), 0)
    right_image = cv2.imread(os.path.join(path, 'right', right_images[0]), 0)

    Kl, Dl, Rl, Pl, Kr, Dr, Rr, Pr, R, T = rectify_matrix.load_camera_params(yaml_file=yaml_file)

    rect_left, rect_right = rectify_matrix.remap_rect_images(left_image, right_image, Kl, Dl, Rl, Pl, Kr, Dr, Rr, Pr)
    debugger.show_stereo_images(rect_left, rect_right, 'rect')

    # Create binary mask
    mask_left, mask_right = debugger.mask_images(rect_left, rect_right, thres=190)
    # debugger.show_stereo_images(mask_left, mask_right, 'mask')

    # Apply mask on original image
    left_image = cv2.bitwise_and(rect_left, rect_left, mask=mask_left)
    right_image = cv2.bitwise_and(rect_right, rect_right, mask=mask_right)

    debugger.show_stereo_images(left_image, right_image, 'Image')
    # Points from Blob
    # point_left = detect_dots(rect_left)
    # point_right = detect_dots(rect_right)
    # # Debug what points where found on each image
    # debug_left, debug_right = draw_matched_points(rect_left, rect_right, point_left, point_right)
    # debugger.show_stereo_images(debug_left, debug_right, 'Blob points')

    # # Matched points
    # m_p_left, m_p_right = match_points(point_left, point_right, 100)
    #
    # left_dp, right_dp = draw_matched_points(rect_left, rect_right, m_p_left, m_p_right, color=(0, 255, 0), radius=10)
    # debugger.show_stereo_images(left_dp, right_dp)

    c_left, contours_left = detect_dots_findcontours(mask_left)
    c_right, contours_right = detect_dots_findcontours(mask_right)
    c_img_left = cv2.drawContours(cv2.cvtColor(mask_left, cv2.COLOR_GRAY2BGR), contours_left, -1, (0, 255, 0), 3)
    c_img_right = cv2.drawContours(cv2.cvtColor(mask_right, cv2.COLOR_GRAY2BGR), contours_right, -1, (0, 255, 0), 3)
    debugger.show_stereo_images(c_img_left, c_img_right, 'contours')
    mcp_left, mcp_right = match_points2(c_left, c_right, threshold=200)

    mc_left, mc_right = draw_matched_points(left_image,right_image, mcp_left, mcp_right, color=(0, 255, 255), radius=5,
                                            thickness=2)
    # cv2.imshow('right', mc_right)
    # cv2.resizeWindow('right', 800, 600)
    # cv2.waitKey(0)
    debugger.show_stereo_images(mc_left, mc_right, 'MatchPoints FC')
    print('wait')

if __name__ == '__main__':
    main()
