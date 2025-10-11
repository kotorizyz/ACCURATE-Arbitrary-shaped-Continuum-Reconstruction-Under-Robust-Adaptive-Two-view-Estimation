import open3d as o3d
import os
import json
import numpy as np
import cv2
from skimage.morphology import skeletonize

dataset_path = 'dataset/dataset_3/'
files = sorted(os.listdir(dataset_path))

def world_to_pixel(points_world, intrinsic, extrinsic):

    intrinsic = np.array([[1988, 0, 256],[0, 1988, 1024],[0,0,1]])

    ones = np.ones((points_world.shape[0], 1))
    points_hom = np.hstack([points_world, ones])  # (N, 4)

    # 世界 -> 相机
    points_cam = (extrinsic @ points_hom.T).T     # (N, 3)
    
    # 相机 -> 像素
    pixels_hom = (intrinsic @ points_cam.T).T     # (N, 3)

    u = pixels_hom[:, 0] / pixels_hom[:, 2]
    v = pixels_hom[:, 1] / pixels_hom[:, 2]

    pixels = np.stack([u, v], axis=1)
    return pixels

for file in files:
    file_path = dataset_path + file

    # camera parameter
    para = json.load(open(file_path + '/camera_params.json'))
    intrinsic_cam1 = np.array(para["cam1"]["intrinsic"])            # 3 * 3
    extrinsic_cam1 = np.array(para["cam1"]["extrinsic"])[:3]        # 3 * 4
    intrinsic_cam2 = np.array(para["cam2"]["intrinsic"])            # 3 * 3
    extrinsic_cam2 = np.array(para["cam2"]["extrinsic"])[:3]        # 3 * 4

    # 3d point cloud
    point_3d = o3d.io.read_point_cloud(file_path + '/curve_points.ply')
    point_3d = np.asarray(point_3d.points)
    N = point_3d.shape[0]

    print(point_3d)
    quit()

    # mask
    mask1 = skeletonize(cv2.imread(file_path + '/mask_cam1.png', cv2.IMREAD_GRAYSCALE) > 10).astype(np.uint8)
    mask2 = skeletonize(cv2.imread(file_path + '/mask_cam2.png', cv2.IMREAD_GRAYSCALE) > 10).astype(np.uint8)
    if(mask1.shape != mask2.shape):
        print('Camera Parameter Error')
        quit()
    image_size = mask1.shape

    # project back
    uv_cam1 = world_to_pixel(point_3d, intrinsic_cam1, extrinsic_cam1)
    uv_cam2 = world_to_pixel(point_3d, intrinsic_cam2, extrinsic_cam2)
    
    img1 = np.zeros(image_size).astype(np.uint8)
    u1 = uv_cam1[:,0].astype(np.int32)
    v1 = uv_cam1[:,1].astype(np.int32)
    img1[v1, u1] = 155
    
    img = mask1 * 100 + img1
    cv2.imwrite('img1.png', img)
    quit()
