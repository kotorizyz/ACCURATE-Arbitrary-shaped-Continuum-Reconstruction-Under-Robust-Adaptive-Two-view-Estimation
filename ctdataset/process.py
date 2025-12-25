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
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,   1]
    ])

    P = K @ extrinsic
    return P

def project_points(points_xyz, P):
    N = points_xyz.shape[0]
    pts_h = np.hstack([points_xyz, np.ones((N,1))])
    proj = (P @ pts_h.T).T

    u = proj[:, 0] / proj[:, 2]
    v = proj[:, 1] / proj[:, 2]
    return np.stack([u, v], axis=1)


file = 'ctdataset/SE5/IM1'
dcm = pydicom.dcmread(file)

num_frames = dcm.NumberOfFrames
num_mats = len(dcm[0x00211009].value)


file_off = 'ctdataset/SE0/IM0'
dcm_off  = pydicom.dcmread(file_off)
mat_off = np.array(dcm_off[0x00289520].value).reshape(4,4)[:3,3]
# mat_off = np.array([-58.5, -68.599998, 690.5])
mat_off = np.array([-57.3, -68.599998, 690.5])

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
for item in dcm[0x0021100b]:
    vals = list(map(float, item[0x0021100c].value))
    P = np.array(vals).reshape(3,4)
    P[:3, :3] *= 10.0
    P = P[:,[1,0,2,3]]
    P_list.append(P)
P_list = np.array(P_list[3:294])

# S = get_source_positions(P_list)[:,[1,0,2]]
# print(fit_rotation_circle(S))
# quit()

# S_proj = np.zeros((num_frames, 3))
# for i, P in enumerate(P_list):
#     C = camera_center_from_P(P)
#     S_proj[i] = C
# S_proj = S_proj[:, [1,0,2]] * 0.1

imgs = dcm.pixel_array
points3D = o3d.io.read_point_cloud('ctdataset/SE6/Segment_1.ply')
# points3D = o3d.io.read_point_cloud('ctdataset/Segment.ply')
points = np.asarray(points3D.points)  # (N, 3)
# print(points)
points += mat_off[None, :]



H, W = 500, 500
for i in [0,145,290]:
    img = imgs[i]

    # if angle2[i] < -90 or angle2[i] > 90:
    #     continue
    # P = calculate_projection_matrix(angle1[i], angle2[i], angle3[i])
    # point2D = project_points(points, P) / 2

    P = P_list[i]
    point2D = project_points(points, P) / 2

    mask = (
        (point2D[:,0] >= 0) & (point2D[:,0] < W) &
        (point2D[:,1] >= 0) & (point2D[:,1] < H)
    )
    point2D = point2D[mask]

    gt_img = img
    gt_img = (gt_img - gt_img.min()) / (gt_img.max() - gt_img.min()) * 255.0
    for point2d in point2D:
        if point2d[0] >=0 and point2d[0] <= W - 1 and point2d[1] >= 1 and point2d[1] <= H:
            gt_img[H - point2d[1].astype(np.int32), point2d[0].astype(np.int32)] = 255
    cv2.imwrite(f'tmp/gt_{i}.png', gt_img)

# points = np.array([[0,0,0]])
# for P in P_list:
#     print(project_points(points, P) / 2)
# quit()
# points[:,0] *= -1
# points[:,1] *= -1