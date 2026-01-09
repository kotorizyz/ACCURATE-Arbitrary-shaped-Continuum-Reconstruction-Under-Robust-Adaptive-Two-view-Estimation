import pydicom
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from pydicom.tag import Tag
import cv2
from pydicom.pixel_data_handlers.util import apply_voi_lut
from skimage.morphology import skeletonize
import open3d as o3d
from scipy.linalg import rq

def calculate_projection_matrix(
    angle1_deg,
    angle2_deg,
    angle3_deg,
    SID=1195.0,
    SOD=720.0,
    pixel_spacing=(0.4, 0.4),
    image_size=(1000, 1000)
):

    a1 = np.deg2rad(angle1_deg)
    a2 = np.deg2rad(angle2_deg)
    a3 = np.deg2rad(angle3_deg)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(a1), -np.sin(a1)],
        [0, np.sin(a1),  np.cos(a1)]
    ])

    Ry = np.array([
        [ np.cos(a3), 0, np.sin(a3)],
        [0, 1, 0],
        [-np.sin(a3), 0, np.cos(a3)]
    ])

    Rz = np.array([
        [np.cos(a2), -np.sin(a2), 0],
        [np.sin(a2),  np.cos(a2), 0],
        [0, 0, 1]
    ])

    R = Rz @ Ry @ Rx

    S0 = np.array([0.0, SOD, 0])
    S = R @ S0

    zc = -S / np.linalg.norm(S)
    xc = np.array([1.0, 0.0, 0.0])
    yc = np.cross(zc, xc)
    yc /= np.linalg.norm(yc)
    xc = np.cross(yc, zc)

    Rc = np.vstack([xc, yc, zc])

    t = -Rc @ S.reshape(3, 1)
    extrinsic = np.hstack([Rc, t])

    dx, dy = pixel_spacing
    fx = SID / dx
    fy = SID / dy

    cx = image_size[0] / 2.0
    cy = image_size[1] / 2.0

    K = np.array([
        [fx/2, 0,  cx/2],
        [0,  -fy/2, cy/2],
        [0,  0,   1]
    ])

    P = K @ extrinsic
    return P, K, extrinsic

def project_points(points_xyz, P):
    N = points_xyz.shape[0]
    pts_h = np.hstack([points_xyz, np.ones((N,1))])
    proj = (P @ pts_h.T).T

    u = proj[:, 0] / proj[:, 2]
    v = proj[:, 1] / proj[:, 2]
    return np.stack([u, v], axis=1)

# 0,1,2,3,4
idx_guidewire = 0

if idx_guidewire == 0:
    file = f'ctdataset/SE5/IM1'
elif idx_guidewire == 1:
    file = f'ctdataset/SE5/IM13'
elif idx_guidewire == 2:
    file = f'ctdataset/SE5/IM21'
elif idx_guidewire == 3:
    file = f'ctdataset/SE5/IM28'
elif idx_guidewire == 4:
    file = f'ctdataset/SE5/IM35'
dcm = pydicom.dcmread(file)
num_frames = dcm.NumberOfFrames
num_mats = len(dcm[0x00211009].value)

file_off = f'ctdataset/SE{idx_guidewire}/IM0'
dcm_off  = pydicom.dcmread(file_off)
mat_off = np.array(dcm_off[0x0289520].value)[[3,7,11]]

print(num_frames, num_mats, mat_off)

angle1 = np.array([dcm[0x00191001].value]*num_frames) + dcm[0x00191197].value
angle2 = np.array([dcm[0x00191002].value]*num_frames) + dcm[0x00191198].value
angle3 = np.array([dcm[0x00191003].value]*num_frames) + dcm[0x00191199].value
dlt_angle1 = (angle1[-1] - angle1[0]) / (num_frames - 1)
dlt_angle2 = (angle2[-1] - angle2[0]) / (num_frames - 1)
dlt_angle3 = (angle3[-1] - angle3[0]) / (num_frames - 1)
for i in range(num_frames):
    angle1[i] = angle1[0] + i * dlt_angle1
    angle2[i] = angle2[0] + i * dlt_angle2
    angle3[i] = angle3[0] + i * dlt_angle3

P_list = []

# for i in range(len(angle1)):
#     P = calculate_projection_matrix(angle1[i], angle2[2], angle3[i])
#     P[0:2] /= 2
#     P_list.append(P)

# for item in dcm[0x0021100b]:
#     vals = list(map(float, item[0x0021100c].value))
#     P = np.array(vals).reshape(3,4)
#     P[:3, :3] *= 10.0
#     P = P[:,[1,0,2,3]]
#     P[0:2] /= 2
#     P_list.append(P)
# P_list = np.array(P_list[3:294])

imgs = dcm.pixel_array
points3D = o3d.io.read_point_cloud(f'ctdataset/SE6/Segment_{idx_guidewire}.ply')
points = np.asarray(points3D.points)  # (N, 3)
points += mat_off[None, :]

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points)
# o3d.io.write_point_cloud(f"output/real/gt_{idx_guidewire}.ply", pcd)
# quit()

H, W = 500, 500
for i in range(num_frames):
    img = imgs[i]

    if angle2[i] < -90 or angle2[i] > 90:
        continue
    P, K, Rt = calculate_projection_matrix(angle1[i], angle2[i], angle3[i])
    np.save(f'tmp/K_{i}.npy', K)
    np.save(f'tmp/Rt_{i}.npy', Rt)
    point2D = project_points(points, P)

    # P = P_list[i]
    # point2D = project_points(points, P)

    mask = (
        (point2D[:,0] >= 0) & (point2D[:,0] < W) &
        (point2D[:,1] >= 0) & (point2D[:,1] < H)
    )
    point2D = point2D[mask]

    gt_img = img
    gt_img = (gt_img - gt_img.min()) / (gt_img.max() - gt_img.min()) * 255.0
    cv2.imwrite(f'tmp/gt_{i}.png', gt_img)

    for point2d in point2D:
        if point2d[0] >=0 and point2d[0] <= W - 1 and point2d[1] >= 1 and point2d[1] <= H:
            gt_img[point2d[1].astype(np.int32), point2d[0].astype(np.int32)] = 255
    cv2.imwrite(f'tmp/fusion_{i}.png', gt_img)

    mask_img = np.zeros((H,W))
    for point2d in point2D:
        if point2d[0] >=0 and point2d[0] <= W - 1 and point2d[1] >= 1 and point2d[1] <= H:
            mask_img[point2d[1].astype(np.int32), point2d[0].astype(np.int32)] = 1
    np.save(f'tmp/mask_{i}.npy', mask_img)

cv2.imwrite('check.png', mask_img*255)