import cv2
import numpy as np
import os

PATH = '/home/daniel/Pictures/image_logs/SM2'
ORIGINAL_IMG = '20250801'
OUTPUT_IMG = '20250801_resized_612_512'
camera = ['left', 'right']
if __name__ == '__main__':
    for side in camera:
        input_imgs = os.listdir(os.path.join(PATH, ORIGINAL_IMG, side))
        os.makedirs(os.path.join(PATH, OUTPUT_IMG, side), exist_ok=True)
        for img in input_imgs:
            image = cv2.imread(os.path.join(PATH, ORIGINAL_IMG, side, img),cv2.IMREAD_COLOR)
            # resized_img = cv2.resize(image, (1224,1024), cv2.INTER_LINEAR)
            resized_img = cv2.resize(image, (612,512), cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(PATH, OUTPUT_IMG, side, img), resized_img)