import open3d as o3d
import os
import json
import numpy as np
import cv2
from skimage.morphology import skeletonize

dataset_path = 'dataset/dataset_3/'
files = sorted(os.listdir(dataset_path))

def get_projection_matrix(cam):
    K = np.array(cam["intrinsic"])
    extrinsic = np.array(cam["extrinsic"])
    Rt = extrinsic[:3, :]
    P = K @ Rt
    return P

def project_point(P, Xw):
    x = P @ Xw
    u = x[0] / x[2]
    v = x[1] / x[2]
    return np.array([u, v])

def project_points(P, X_h):
    # (3x4) @ (4xN) -> (3xN)
    x = (P @ X_h.T)
    x[:2, :] /= x[2, :]
    return x[:2, :].T

for file in files:
    file_path = dataset_path + file

    # camera parameter
    para = json.load(open(file_path + '/camera_params.json'))
    P1 = get_projection_matrix(para["cam1"])
    P2 = get_projection_matrix(para["cam2"])

    # 3d point cloud
    point_3d = o3d.io.read_point_cloud(file_path + '/curve_points.ply')
    point_3d = np.asarray(point_3d.points)
    N = point_3d.shape[0]

    # mask
    mask1 = skeletonize(cv2.imread(file_path + '/mask_cam1.png', cv2.IMREAD_GRAYSCALE) > 10).astype(np.uint8)
    mask2 = skeletonize(cv2.imread(file_path + '/mask_cam2.png', cv2.IMREAD_GRAYSCALE) > 10).astype(np.uint8)
    if(mask1.shape != mask2.shape):
        print('Camera Parameter Error')
        quit()
    image_size = mask1.shape

    # project back
    point_3d_h = np.hstack([point_3d, np.ones((N, 1))])
    point_2d_cam1 = project_points(P1, point_3d_h).astype(np.int64)
    point_2d_cam2 = project_points(P2, point_3d_h).astype(np.int64)
    
    img1 = np.zeros(image_size).astype(np.uint8)
    u1 = point_2d_cam1[:,0]
    v1 = point_2d_cam1[:,1]
    img1[v1, u1] = 255
    
    # img = mask1 * 100 + img1
    cv2.imwrite('img1.png', img1)
    quit()
