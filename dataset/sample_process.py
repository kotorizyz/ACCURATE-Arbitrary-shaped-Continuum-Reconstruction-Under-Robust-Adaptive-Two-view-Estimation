import open3d as o3d
import os
import json
import numpy as np
import cv2
from skimage.morphology import skeletonize
import torch

dataset_path = 'dataset/dataset_3/'
files = sorted(os.listdir(dataset_path))

def world_to_pixel(points_world, intrinsic, extrinsic):

    # intrinsic = np.array([[1988, 0, 256],[0, 1988, 1024],[0,0,1]])

    ones = np.ones((points_world.shape[0], 1))
    points_hom = np.hstack([points_world, ones])  # (N, 4)

    points_cam = (extrinsic @ points_hom.T).T     # (N, 3)
    pixels_hom = (intrinsic @ points_cam.T).T     # (N, 3)

    u = pixels_hom[:, 0] / pixels_hom[:, 2]
    v = pixels_hom[:, 1] / pixels_hom[:, 2]

    pixels = np.stack([u, v], axis=1)
    return pixels

for file in files:
    file_idx = int(file.split('_')[-1])
    file_path = dataset_path + file

    # camera parameter
    para = json.load(open(file_path + '/camera_params.json'))
    
    # intrinsic_cam1 = np.array(para["cam1"]["intrinsic"])              # (3, 3)
    intrinsic_cam1 = np.array([[1988, 0, 256],[0, 1988, 1024],[0,0,1]])
    extrinsic_cam1 = np.array(para["cam1"]["extrinsic"])[:3]            # (3, 4)

    # intrinsic_cam2 = np.array(para["cam2"]["intrinsic"])              # (3, 3)
    intrinsic_cam2 = np.array([[1988, 0, 256],[0, 1988, 1024],[0,0,1]])
    extrinsic_cam2 = np.array(para["cam2"]["extrinsic"])[:3]            # (3, 4)

    # 3d point cloud
    point_3d = o3d.io.read_point_cloud(file_path + '/curve_points.ply')
    point_3d = np.asarray(point_3d.points)
    N = point_3d.shape[0]
    
    image_size = (2048, 512)

    # project back
    uv_cam1 = world_to_pixel(point_3d, intrinsic_cam1, extrinsic_cam1)
    uv_cam2 = world_to_pixel(point_3d, intrinsic_cam2, extrinsic_cam2)

    img1 = np.zeros(image_size).astype(np.uint8)
    u1 = uv_cam1[:,0].astype(np.int32)
    v1 = uv_cam1[:,1].astype(np.int32)
    img1[v1, u1] = 1

    img2 = np.zeros(image_size).astype(np.uint8)
    u2 = uv_cam2[:,0].astype(np.int32)
    v2 = uv_cam2[:,1].astype(np.int32)
    img2[v2, u2] = 1
    
    # M_ij Mat
    x1 = np.argwhere(img1 > 0)                                  # (N1, 2)
    x2 = np.argwhere(img2 > 0)                                  # (N2, 2)
    M_ij = np.zeros((x1.shape[0], x2.shape[0])).astype(np.uint8)   # (N1, N2)
    for i in range(N):
        u1 = uv_cam1[i,0].astype(np.int32)
        v1 = uv_cam1[i,1].astype(np.int32)
        u2 = uv_cam2[i,0].astype(np.int32)
        v2 = uv_cam2[i,1].astype(np.int32)
        idx1 = np.where((x1[:,0]==v1) & (x1[:,1]==u1))[0][0]
        idx2 = np.where((x2[:,0]==v2) & (x2[:,1]==u2))[0][0]
        M_ij[idx1, idx2] = 1

    # img = mask1 * 100 + img1
    # cv2.imwrite('img1.png', img1)
    # quit()

    # store data
    uv1 = torch.tensor(uv_cam1, dtype=torch.float32)            # (N, 2)
    uv2 = torch.tensor(uv_cam2, dtype=torch.float32)            # (N, 2)
    points = torch.tensor(point_3d, dtype=torch.float32)        # (N, 3)
    K1 = torch.tensor(intrinsic_cam1, dtype=torch.float32)      # (3, 3)
    RT1 = torch.tensor(extrinsic_cam1, dtype=torch.float32)     # (3, 4)
    K2 = torch.tensor(intrinsic_cam2, dtype=torch.float32)      # (3, 3)
    RT2 = torch.tensor(extrinsic_cam2, dtype=torch.float32)     # (3, 4)
    image_size = torch.tensor(image_size, dtype=torch.int32)    # (2, )
    x1 = torch.tensor(x1, dtype=torch.int32)                    # (N1, 2)
    x2 = torch.tensor(x2, dtype=torch.int32)                    # (N2, 2)
    M_ij = torch.tensor(M_ij, dtype=torch.uint8)                # (N1, N2)
    data = {
        'uv1': uv1,
        'uv2': uv2,
        'points': points,
        'K1': K1,
        'K2': K2,
        'RT1': RT1,
        'RT2': RT2,
        'image_size': image_size,
        'x1' : x1,
        'x2' : x2,
        'M_ij' : M_ij
    }
    torch.save(data, f'dataset/processed_data/data_{file_idx}.pt')