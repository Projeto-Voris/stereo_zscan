import cv2
import numpy as np
import glob
import open3d as o3d
import matplotlib.pyplot as plt
import yaml
import os
import rectify_matrix



def rectify_images(imgL, imgR, mtxL, distL, R1, P1, mtxR, distR, R2, P2):
    """Retifica imagens das câmeras esquerda e direita."""
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    left_map1, left_map2 = cv2.initUndistortRectifyMap(mtxL, distL, R1, P1, grayL.shape[::-1], cv2.CV_16SC2)
    right_map1, right_map2 = cv2.initUndistortRectifyMap(mtxR, distR, R2, P2, grayR.shape[::-1], cv2.CV_16SC2)
    rectifiedL = cv2.remap(grayL, left_map1, left_map2, cv2.INTER_LINEAR)
    rectifiedR = cv2.remap(grayR, right_map1, right_map2, cv2.INTER_LINEAR)
    return rectifiedL, rectifiedR


def calculate_disparity(rectifiedL, rectifiedR):
    """Calcula a disparidade entre as imagens retificadas."""
    stereo = cv2.StereoBM_create(numDisparities=16 * 10, blockSize=15)
    disparity = stereo.compute(rectifiedL, rectifiedR)
    return disparity


def generate_point_cloud(disparity, Q, imgL):
    """Gera a nuvem de pontos 3D a partir da disparidade."""
    points_3D = cv2.reprojectImageTo3D(disparity, Q)
    mask = disparity > disparity.min()
    output_points = points_3D[mask]
    output_colors = imgL[mask]
    return output_points, output_colors


def save_point_cloud(points, colors, output_file, scale_factor=0.00345):
    """Salva a nuvem de pontos em um arquivo PLY."""
    points_scaled = points * scale_factor
    ply_header = '''ply
                    format ascii 1.0
                    element vertex %(vert_num)d
                    property float x
                    property float y
                    property float z
                    property uchar red
                    property uchar green
                    property uchar blue
                    end_header
                    '''
    with open(output_file, 'w') as f:
        f.write(ply_header % dict(vert_num=len(points_scaled)))
        for point, color in zip(points_scaled, colors):
            f.write(f"{point[0]} {point[1]} {point[2]} {color[2]} {color[1]} {color[0]}\n")


def visualize_point_cloud(points, colors):
    """Visualiza a nuvem de pontos usando Open3D e Matplotlib."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    o3d.visualization.draw_geometries([pcd])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors / 255.0, marker='.')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def show_image(image):
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyWindow('Image')

def show_stereo_images(imgR, imgL):
    img_concatenate = np.concatenate((imgL, imgR), axis=1)
    cv2.namedWindow('Stereo', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Stereo', int(img_concatenate.shape[1] / 4), int(img_concatenate.shape[0] / 4))
    cv2.imshow('Stereo', img_concatenate)
    cv2.waitKey(0)
    cv2.destroyWindow('Stereo')

def main():
    # Caminho para o arquivo de parâmetros
    yaml_file = 'cfg/20240815_rect_1.yaml'

    image_dir = '/home/daniel/PycharmProjects/stereo_active/images/SM3-20240820 - RRP'
    left_images = sorted(os.listdir(os.path.join(image_dir, 'left')))
    right_images = sorted(os.listdir(os.path.join(image_dir, 'right')))
    image_files_left = []
    image_files_right = []
    for (left,right) in zip(left_images, right_images):
        image_files_left = sorted(os.path.join(image_dir, 'left', left))
        image_files_right = sorted(os.path.join(image_dir, 'right', right))

    # Carrega os parâmetros do sistema estéreo
    mtxL, distL, R1, P1, mtxR, distR, R2, P2, Q = rectify_matrix.load_camera_params(yaml_file)

    all_points, all_colors = [], []

    for imgL_path, imgR_path in zip(image_files_left, image_files_right):
        imgL = cv2.imread(imgL_path)
        imgR = cv2.imread(imgR_path)
        show_stereo_images(imgR, imgL)
        rectifiedL, rectifiedR = rectify_images(imgL, imgR, mtxL, distL, R1, P1, mtxR, distR, R2, P2)
        show_stereo_images(rectifiedR, rectifiedL)
        disparity = calculate_disparity(rectifiedL, rectifiedR)
        output_points, output_colors = generate_point_cloud(disparity, Q, imgL)
        print('wait')
        all_points.append(output_points)
        all_colors.append(output_colors)

    all_points = np.concatenate(all_points, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)

    # save_point_cloud(all_points, all_colors, 'combined_output_points_mm.ply')
    visualize_point_cloud(all_points, all_colors)


if __name__ == "__main__":
    main()
