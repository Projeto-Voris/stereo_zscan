import cv2
import numpy as np
import os
import cupy as cp
import gc

from extras.debugger import plot_points_on_image, show_stereo_images
from include.InverseTriangulation import InverseTriangulation
import extras.project_points as project_points
def main():
    # Paths for yaml file and images
    yaml_file = 'cfg/SM3-20250424.yaml'
    # images_path = 'images/SM4-20241004 -calib 25x25'
    images_path = '/home/daniel/Insync/daniel.regner@labmetro.ufsc.br/Google Drive - Shared drives/VORIS - Media/Experimentos/SM3 - Padrão aleatório/Calibração/SM3 -20250424 - 10x10'
    left_imgs_list = sorted(os.listdir(os.path.join(images_path, 'left')))

    right_imgs_list = sorted(os.listdir(os.path.join(images_path, 'right')))
    # images_path = '/home/daniel/Pictures/sm3'
    n_img = 10
    Zscan = InverseTriangulation(yaml_file=yaml_file)
    # # Identify all images from path file
    left_imgs = Zscan.read_images(path=os.path.join(images_path,'left'), images_list=left_imgs_list, n_imgs=n_img)
    right_imgs = Zscan.read_images(path=os.path.join(images_path,'right'), images_list=right_imgs_list, n_imgs=n_img)
    Zscan.convert_images(left_imgs=left_imgs, right_imgs=right_imgs, apply_clahe=True, undist=True)

    points3d = Zscan.points3d(x_lim=(-180,300), y_lim=(-140,300), z_lim=(0,10), xy_step=10, z_step=1, visualize=False)
    uv_left = Zscan.transform_gcs2ccs(points_3d=points3d, cam_name='left')
    uv_right = Zscan.transform_gcs2ccs(points_3d=points3d, cam_name='right')
    output_image_L = plot_points_on_image(image=cp.asnumpy(Zscan.left_images[:, :, 0]), points=uv_left, color=(0, 255, 0),
                                                   radius=5,
                                                   thickness=1)
    output_image_R = plot_points_on_image(image=cp.asnumpy(Zscan.right_images[:, :, 0]), points=uv_right, color=(0, 255, 0),
                                                   radius=5,
                                                   thickness=1)

    show_stereo_images(output_image_L, output_image_R, "Remaped points")
    cv2.waitKey(0)
    # print('wait')


if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()