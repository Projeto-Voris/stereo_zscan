import cv2
import numpy as np
import sys
import os
import yaml

class StereoOptimizerBase:
    def __init__(self, yaml_file, num_disparities_range=(16, 254, 16), block_size_range=(3, 31, 2)):
        self.num_disparities_range = num_disparities_range
        self.block_size_range = block_size_range
        self.camera_params = {
            'left': {'kk': np.array([]), 'kc': np.array([]), 'r': np.array([]), 'p': np.array([])},
            'right': {'kk': np.array([]), 'kc': np.array([]), 'r': np.array([]), 'p': np.array([])},
            'stereo': {'R': np.array([]), 'T': np.array([])}
        }
        self.read_yaml_file(yaml_file)

    def read_yaml_file(self, yaml_file):
        with open(yaml_file) as file:
            params = yaml.safe_load(file)
        self.camera_params['left']['kk'] = np.array(params['camera_matrix_left'], dtype=np.float64)
        self.camera_params['left']['kc'] = np.array(params['dist_coeffs_left'], dtype=np.float64)
        self.camera_params['left']['r'] = np.array(params['rot_matrix_left'], dtype=np.float64)
        self.camera_params['left']['p'] = np.array(params['proj_matrix_left'], dtype=np.float64)
        self.camera_params['right']['kk'] = np.array(params['camera_matrix_right'], dtype=np.float64)
        self.camera_params['right']['kc'] = np.array(params['dist_coeffs_right'], dtype=np.float64)
        self.camera_params['right']['r'] = np.array(params['rot_matrix_right'], dtype=np.float64)
        self.camera_params['right']['p'] = np.array(params['proj_matrix_right'], dtype=np.float64)
        self.camera_params['stereo']['R'] = np.array(params['R'], dtype=np.float64)
        self.camera_params['stereo']['T'] = np.array(params['T'], dtype=np.float64)

    def rectify_images(self, imgL, imgR):
        K1 = self.camera_params['left']['kk']
        D1 = self.camera_params['left']['kc']
        R1 = self.camera_params['left']['r']
        P1 = self.camera_params['left']['p']
        K2 = self.camera_params['right']['kk']
        D2 = self.camera_params['right']['kc']
        R2 = self.camera_params['right']['r']
        P2 = self.camera_params['right']['p']
        image_size = (imgL.shape[1], imgL.shape[0])
        map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_16SC2)
        map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_16SC2)
        rectifiedL = cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR)
        rectifiedR = cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR)
        return rectifiedL, rectifiedR, (map1x, map1y, map2x, map2y)

    def score_disparity(self, disparity, mask=None):
        if mask is not None:
            return np.var(disparity[mask > 0])
        return np.var(disparity)


class StereoSGBMOptimizer(StereoOptimizerBase):
    def compute_sgbm_disparity(self, imgL, imgR, num_disparities, block_size, params=None):
        min_disparity = params.get('minDisparity', 0) if params else 0
        P1 = params.get('P1', 8 * 1 * block_size ** 2) if params else 8 * 1 * block_size ** 2
        P2 = params.get('P2', 32 * 1 * block_size ** 2) if params else 32 * 1 * block_size ** 2
        disp12MaxDiff = params.get('disp12MaxDiff', 1) if params else 1
        uniquenessRatio = params.get('uniquenessRatio', 10) if params else 10
        speckleWindowSize = params.get('speckleWindowSize', 100) if params else 100
        speckleRange = params.get('speckleRange', 32) if params else 32
        preFilterCap = params.get('preFilterCap', 63) if params else 63
        mode = params.get('mode', cv2.STEREO_SGBM_MODE_SGBM) if params else cv2.STEREO_SGBM_MODE_SGBM

        stereo = cv2.StereoSGBM_create(
            minDisparity=min_disparity,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=P1,
            P2=P2,
            disp12MaxDiff=disp12MaxDiff,
            uniquenessRatio=uniquenessRatio,
            speckleWindowSize=speckleWindowSize,
            speckleRange=speckleRange,
            preFilterCap=preFilterCap,
            mode=mode
        )
        disparity = stereo.compute(imgL, imgR)
        return disparity

    def show_trackbar_optimization(self, imgL, imgR, mask=None, window_name='StereoSGBM Tuner'):
        def nothing(x): pass
        def save_params_to_yaml(params, filename='sgbm_params.yaml'):
            with open(filename, 'w') as f:
                yaml.dump(params, f)
            print(f"Parameters saved to {filename}")

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1600, 600)
        max_num_disp = self.num_disparities_range[1]
        min_num_disp = self.num_disparities_range[0]
        step_num_disp = self.num_disparities_range[2]
        max_block_size = self.block_size_range[1]
        min_block_size = self.block_size_range[0]
        step_block_size = self.block_size_range[2]

        cv2.createTrackbar('numDisparities', window_name, min_num_disp // step_num_disp, (max_num_disp - min_num_disp) // step_num_disp, nothing)
        cv2.createTrackbar('blockSize', window_name, (min_block_size - 5) // 2, (max_block_size - 5) // 2, nothing)
        cv2.createTrackbar('P1', window_name, 8, 100, nothing)
        cv2.createTrackbar('P2', window_name, 32, 200, nothing)
        cv2.createTrackbar('preFilterCap', window_name, 31, 62, nothing)
        cv2.createTrackbar('uniquenessRatio', window_name, 15, 100, nothing)
        cv2.createTrackbar('speckleWindowSize', window_name, 0, 200, nothing)
        cv2.createTrackbar('speckleRange', window_name, 2, 32, nothing)
        cv2.createTrackbar('disp12MaxDiff', window_name, 1, 25, nothing)
        cv2.createTrackbar('lambda', window_name, 6000, 10000, nothing)
        cv2.createTrackbar('sigmaColor', window_name, 1, 3, nothing)

        info_text = "Press 's' to save params, ESC to exit"
        matcher_left = matcher_right = wls_filter = None

        while True:
            num_disp = cv2.getTrackbarPos('numDisparities', window_name) * step_num_disp
            num_disp = max(step_num_disp, num_disp)
            if num_disp % 16 != 0:
                num_disp += 16 - (num_disp % 16)
            block_size = 5 + 2 * cv2.getTrackbarPos('blockSize', window_name)
            if block_size % 2 == 0:
                block_size += 1
            if block_size < 5:
                block_size = 5

            params = {
                'preFilterCap': cv2.getTrackbarPos('preFilterCap', window_name),
                'uniquenessRatio': cv2.getTrackbarPos('uniquenessRatio', window_name),
                'speckleWindowSize': cv2.getTrackbarPos('speckleWindowSize', window_name),
                'speckleRange': cv2.getTrackbarPos('speckleRange', window_name),
                'disp12MaxDiff': cv2.getTrackbarPos('disp12MaxDiff', window_name),
                'P1': cv2.getTrackbarPos('P1', window_name),
                'P2': cv2.getTrackbarPos('P2', window_name),
                'mode': cv2.STEREO_SGBM_MODE_SGBM,
                'numDisparities': num_disp,
                'blockSize': block_size,
                'lambda': cv2.getTrackbarPos('lambda', window_name),
                'sigmaColor': cv2.getTrackbarPos('sigmaColor', window_name)
            }

            matcher_left = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=num_disp,
                blockSize=block_size,
                P1=params['P1'],
                P2=params['P2'],
                disp12MaxDiff=params['disp12MaxDiff'],
                uniquenessRatio=params['uniquenessRatio'],
                speckleWindowSize=params['speckleWindowSize'],
                speckleRange=params['speckleRange'],
                preFilterCap=params['preFilterCap'],
                mode=params['mode']
            )
            matcher_right = cv2.ximgproc.createRightMatcher(matcher_left)
            if wls_filter is None:
                wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left)
            wls_filter.setLambda(params['lambda'])
            wls_filter.setSigmaColor(params['sigmaColor'])

            disp_left = matcher_left.compute(imgL, imgR).astype(np.int16)
            disp_right = matcher_right.compute(imgR, imgL).astype(np.int16)
            filtered_disp = wls_filter.filter(disp_left, imgL, None, disp_right)

            disp_vis = cv2.normalize(filtered_disp, None, 0, 255, cv2.NORM_MINMAX)
            disp_vis = np.uint8(disp_vis)
            disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
            disp_left_vis = cv2.normalize(disp_left, None, 0, 255, cv2.NORM_MINMAX)
            disp_left_vis = np.uint8(disp_left_vis)
            disp_left_vis = cv2.applyColorMap(disp_left_vis, cv2.COLORMAP_JET)
            combined = np.hstack((disp_left_vis, disp_vis))
            cv2.putText(combined, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow(window_name, combined)

            key = cv2.waitKey(50)
            if key == 27:
                break
            elif key == ord('s'):
                save_params_to_yaml(params)
        cv2.destroyWindow(window_name)


class StereoBMOptimizer(StereoOptimizerBase):
    def compute_disparity(self, imgL, imgR, num_disparities, block_size, params=None):
        stereo = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
        if params:
            stereo.setPreFilterType(params.get('preFilterType', 1))
            stereo.setPreFilterSize(params.get('preFilterSize', 5))
            stereo.setPreFilterCap(params.get('preFilterCap', 31))
            stereo.setTextureThreshold(params.get('textureThreshold', 10))
            stereo.setUniquenessRatio(params.get('uniquenessRatio', 15))
            stereo.setSpeckleWindowSize(params.get('speckleWindowSize', 0))
            stereo.setSpeckleRange(params.get('speckleRange', 2))
            stereo.setDisp12MaxDiff(params.get('disp12MaxDiff', 1))
        disparity = stereo.compute(imgL, imgR)
        return disparity

    def show_trackbar_optimization(self, imgL, imgR, mask=None, window_name='StereoBM Tuner'):
        def nothing(x): pass
        def save_params_to_yaml(params, filename='bm_params.yaml'):
            with open(filename, 'w') as f:
                yaml.dump(params, f)
            print(f"Parameters saved to {filename}")

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1000, 600)

        max_num_disp = self.num_disparities_range[1]
        min_num_disp = self.num_disparities_range[0]
        step_num_disp = self.num_disparities_range[2]
        max_block_size = self.block_size_range[1]
        min_block_size = self.block_size_range[0]
        step_block_size = self.block_size_range[2]

        cv2.createTrackbar('numDisparities', window_name, min_num_disp // step_num_disp, (max_num_disp - min_num_disp) // step_num_disp, nothing)
        cv2.createTrackbar('blockSize', window_name, (min_block_size - 5) // 2, (max_block_size - 5) // 2, nothing)
        cv2.createTrackbar('preFilterType', window_name, 1, 1, nothing)
        cv2.createTrackbar('preFilterSize', window_name, 5, 25, nothing)
        cv2.createTrackbar('preFilterCap', window_name, 31, 62, nothing)
        cv2.createTrackbar('textureThreshold', window_name, 10, 100, nothing)
        cv2.createTrackbar('uniquenessRatio', window_name, 15, 100, nothing)
        cv2.createTrackbar('speckleWindowSize', window_name, 0, 200, nothing)
        cv2.createTrackbar('speckleRange', window_name, 2, 32, nothing)
        cv2.createTrackbar('disp12MaxDiff', window_name, 1, 25, nothing)
        cv2.createTrackbar('lambda', window_name, 8000, 10000, nothing)
        cv2.createTrackbar('sigmaColor', window_name, 1, 3, nothing)

        info_text = "Press 's' to save params, ESC to exit"
        matcher_left = matcher_right = wls_filter = None

        while True:
            num_disp = cv2.getTrackbarPos('numDisparities', window_name) * step_num_disp
            num_disp = max(step_num_disp, num_disp)
            if num_disp % 16 != 0:
                num_disp += 16 - (num_disp % 16)
            block_size = 5 + 2 * cv2.getTrackbarPos('blockSize', window_name)
            if block_size % 2 == 0:
                block_size += 1
            if block_size < 5:
                block_size = 5

            params = {
                'preFilterType': cv2.getTrackbarPos('preFilterType', window_name),
                'preFilterSize': max(5, cv2.getTrackbarPos('preFilterSize', window_name) | 1),
                'preFilterCap': cv2.getTrackbarPos('preFilterCap', window_name),
                'textureThreshold': cv2.getTrackbarPos('textureThreshold', window_name),
                'uniquenessRatio': cv2.getTrackbarPos('uniquenessRatio', window_name),
                'speckleWindowSize': cv2.getTrackbarPos('speckleWindowSize', window_name),
                'speckleRange': cv2.getTrackbarPos('speckleRange', window_name),
                'disp12MaxDiff': cv2.getTrackbarPos('disp12MaxDiff', window_name),
                'numDisparities': num_disp,
                'blockSize': block_size,
                'lambda': cv2.getTrackbarPos('lambda', window_name),
                'sigmaColor': cv2.getTrackbarPos('sigmaColor', window_name)
            }

            matcher_left = cv2.StereoBM_create(numDisparities=num_disp, blockSize=block_size)
            matcher_left.setPreFilterType(params['preFilterType'])
            matcher_left.setPreFilterSize(params['preFilterSize'])
            matcher_left.setPreFilterCap(params['preFilterCap'])
            matcher_left.setTextureThreshold(params['textureThreshold'])
            matcher_left.setUniquenessRatio(params['uniquenessRatio'])
            matcher_left.setSpeckleWindowSize(params['speckleWindowSize'])
            matcher_left.setSpeckleRange(params['speckleRange'])
            matcher_left.setDisp12MaxDiff(params['disp12MaxDiff'])

            matcher_right = cv2.ximgproc.createRightMatcher(matcher_left)
            if wls_filter is None:
                wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left)
            wls_filter.setLambda(params['lambda'])
            wls_filter.setSigmaColor(params['sigmaColor'])

            disp_left = matcher_left.compute(imgL, imgR).astype(np.int16)
            disp_right = matcher_right.compute(imgR, imgL).astype(np.int16)
            filtered_disp = wls_filter.filter(disp_left, imgL, None, disp_right)

            disp_vis = cv2.normalize(filtered_disp, None, 0, 255, cv2.NORM_MINMAX)
            disp_vis = np.uint8(disp_vis)
            disp_vis = cv2.ximgproc.guidedFilter(guide=imgL, src=disp_vis, radius=8, eps=1e-2)
            # mask = disp_vis == 0
            # disp_vis = cv2.inpaint(disp_vis, mask.astype(np.uint8), 5, cv2.INPAINT_TELEA)

            disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
            disp_left_vis = cv2.normalize(disp_left, None, 0, 255, cv2.NORM_MINMAX)
            disp_left_vis = np.uint8(disp_left_vis)
            disp_left_vis = cv2.applyColorMap(disp_left_vis, cv2.COLORMAP_JET)
            combined = np.hstack((disp_left_vis, disp_vis))
            cv2.putText(combined, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow(window_name, combined)

            key = cv2.waitKey(50)
            if key == 27:
                break
            elif key == ord('s'):
                save_params_to_yaml(params)

        cv2.destroyWindow(window_name)




def main():
    path = '/home/daniel/Pictures/SM2_disp'
    yaml_file = '/home/daniel/Pictures/SM2_disp/params.yaml'
    left_images = sorted(os.listdir(os.path.join(path, 'left')))
    right_images = sorted(os.listdir(os.path.join(path, 'right')))

    imgL = cv2.imread(os.path.join(path, 'left', left_images[0]), cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(os.path.join(path, 'right', right_images[0]), cv2.IMREAD_GRAYSCALE)

    if imgL is None or imgR is None:
        print("Error: Could not load input images.")
        sys.exit(1)

    # optimizer_SGBM = StereoSGBMOptimizer(yaml_file=yaml_file)
    # imgL_rect, imgR_rect, _ = optimizer_SGBM.rectify_images(imgL=imgL, imgR=imgR)
    # optimizer_SGBM.show_trackbar_optimization(imgL=imgL_rect, imgR=imgR_rect)
    optimizer_BM = StereoBMOptimizer(yaml_file=yaml_file)
    imgL_rect, imgR_rect, _ = optimizer_BM.rectify_images(imgL=imgL, imgR=imgR)
    optimizer_BM.show_trackbar_optimization(imgL_rect, imgR_rect)

if __name__ == '__main__':
    main()