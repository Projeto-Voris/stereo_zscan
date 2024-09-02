import cv2
import numpy as np
import os

# Scripts to debugg and rectify images
import debugger
import rectify_matrix


def extract_patch(image, point, window_size):
    half_window = window_size // 2
    x, y = point
    patch = image[max(0, y - half_window):min(image.shape[0], y + half_window + 1),
            max(0, x - half_window):min(image.shape[1], x + half_window + 1)]
    return patch


def match_points_with_roi(left_image, right_image, points_left, points_right, window_size=21, threshold=0.2):
    matched_points = []

    for (x_left, y_left) in points_left:
        left_patch = extract_patch(left_image, (x_left, y_left), window_size)

        best_match = None
        best_score = float('inf')

        for (x_right, y_right) in points_right:
            right_patch = extract_patch(right_image, (x_right, y_right), window_size)

            if left_patch.shape == right_patch.shape:
                # Use Sum of Squared Differences (SSD) as a similarity measure
                ssd = np.sum((left_patch.astype('float') - right_patch.astype('float')) ** 2)

                if ssd < best_score and ssd < threshold:
                    best_score = ssd
                    best_match = (x_right, y_right)

        if best_match is not None:
            matched_points.append(((x_left, y_left), best_match))

    return matched_points


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
    matched_left = []
    matched_right = []
    for pL in points_left:
        for pR in points_right:
            if abs(pL[1] - pR[1]) < 5:
                distances = abs((pL[0] - 1397) - (pR[0] - 23))
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
        cv2.circle(image_right, tuple(int(x) for x in pt_right), radius, (0, 255, 0), thickness)
    return image_left, image_right


def draw_matched_points2(image_left, image_right, matched_points, color=(0, 255, 0), radius=5, thickness=2):
    # Convert the images to color if they are grayscale
    if len(image_left.shape) == 2:
        image_left = cv2.cvtColor(image_left, cv2.COLOR_GRAY2BGR)
    if len(image_right.shape) == 2:
        image_right = cv2.cvtColor(image_right, cv2.COLOR_GRAY2BGR)

    # Draw the matched points on both images
    for pt_left, pt_right in matched_points:
        # Draw circles on the left image
        cv2.circle(image_left, tuple(int(x) for x in pt_left), radius, color, thickness)
        # Draw circles on the right image
        cv2.circle(image_right, tuple(int(x) for x in pt_right), radius, (0, 255, 0), thickness)
    return image_left, image_right

def calculate_disparity(rectifiedL, rectifiedR):
    """Calcula a disparidade entre as imagens retificadas."""
    stereo = cv2.StereoBM.create(numDisparities=16 * 10, blockSize=15)
    disparity = stereo.compute(rectifiedL, rectifiedR)
    return disparity

def main():
    path = '/home/daniel/PycharmProjects/stereo_active/images/SM3-20240820 - RRP'
    # path = '/home/daniel/Insync/daniel.regner@labmetro.ufsc.br/Google Drive - Shared drives/VORIS  - Equipe/Sistema de Medição 3 - Stereo Ativo - Projeção Laser/Imagens/Testes/SM3-20240820 - RRP tubo'
    yaml_file = 'cfg/20240815_rect_1.yaml'
    left_images = sorted(os.listdir(os.path.join(path, 'left')))
    right_images = sorted(os.listdir(os.path.join(path, 'right')))

    left_image = cv2.imread(os.path.join(path, 'left', left_images[0]), 0)
    right_image = cv2.imread(os.path.join(path, 'right', right_images[0]), 0)

    Kl, Dl, Rl, Pl, Kr, Dr, Rr, Pr, R, T = rectify_matrix.load_camera_params(yaml_file=yaml_file)

    rect_left, rect_right = rectify_matrix.remap_rect_images(left_image, right_image, Kl, Dl, Rl, Pl, Kr, Dr, Rr, Pr)
    debugger.show_stereo_images(rect_left, rect_right, 'rect')
    disp = calculate_disparity(rect_left, rect_right)
    cv2.namedWindow('disp', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('disp', 640, 480)
    cv2.imshow('disp', disp)
    cv2.waitKey(0)

    # Create binary mask
    mask_left, mask_right = debugger.mask_images(rect_left, rect_right, thres=150)
    debugger.show_stereo_images(mask_left, mask_right, 'mask')

    # Apply mask on original image
    left_image = cv2.bitwise_and(rect_left, rect_left, mask=mask_left)
    right_image = cv2.bitwise_and(rect_right, rect_right, mask=mask_right)

    # debugger.show_stereo_images(left_image, right_image, 'Image')

    c_left, contours_left = detect_dots_findcontours(mask_left)
    c_right, contours_right = detect_dots_findcontours(mask_right)
    c_img_left = cv2.drawContours(cv2.cvtColor(mask_left, cv2.COLOR_GRAY2BGR), contours_left, -1, (0, 255, 0), 3)
    c_img_right = cv2.drawContours(cv2.cvtColor(mask_right, cv2.COLOR_GRAY2BGR), contours_right, -1, (0, 255, 0), 3)
    # debugger.show_stereo_images(c_img_left, c_img_right, 'contours')

    matched_points = match_points_with_roi(left_image, right_image, c_left, c_right, window_size=50, threshold=10)

    mcp_left, mcp_right = match_points(c_left, c_right, threshold=200)

    # mc_left, mc_right = draw_matched_points(left_image, right_image, mcp_left, mcp_right, color=(0, 255, 255), radius=5, thickness=2)
    mc_left, mc_right = draw_matched_points2(left_image, right_image, matched_points,
                                             color=(0, 255, 0), radius=5, thickness=2)
    # cv2.imshow('right', mc_right)
    # cv2.resizeWindow('right', 800, 600)
    # cv2.waitKey(0)
    debugger.show_stereo_images(mc_left, mc_right, 'MatchPoints FC')
    print('wait')


if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
