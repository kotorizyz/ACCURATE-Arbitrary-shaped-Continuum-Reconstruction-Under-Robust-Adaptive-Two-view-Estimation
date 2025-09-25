import cv2
import numpy as np
import json
import open3d as o3d
from skimage.morphology import skeletonize
from scipy.signal import convolve2d

from ordered import ordered_points_from_mask

cnt = np.zeros((3000, 4096))

def parse_camera_dict(cam_dict):
    """
    cam_dict example:
    {'intrinsic': [[...],[...],[...]],
     'extrinsic': [[...],..., [...]],  # 4x4
     'resolution': [H,W] or [W,H]
    }
    Returns: K (3x3), extrinsic_4x4 (4x4), resolution (tuple)
    """
    K = np.array(cam_dict['intrinsic'], dtype=float)
    ext = np.array(cam_dict['extrinsic'], dtype=float)
    res = tuple(cam_dict.get('resolution', (None, None)))
    return K, ext, res

def extrinsic_to_Rt(ext4, convention='world2cam'):
    """
    ext4: 4x4 extrinsic matrix (numpy array)
    convention: 'world2cam' or 'cam2world'
      - 'world2cam': ext4 is [R_wc | t_wc; 0 1], mapping world -> camera: X_cam = R_wc X_world + t_wc
      - 'cam2world': ext4 is [R_cw | t_cw; 0 1], mapping camera -> world: X_world = R_cw X_cam + t_cw
    Returns: R_wc (3x3), t_wc (3x1) representing world->camera
    """
    R = ext4[:3, :3]
    t = ext4[:3, 3].reshape(3,1)
    if convention == 'world2cam':
        return R, t
    elif convention == 'cam2world':
        # convert to world->camera
        R_wc = R.T
        t_wc = - R.T @ t
        return R_wc, t_wc
    else:
        raise ValueError("convention must be 'world2cam' or 'cam2world'")

def get_stereo_params_from_dict(cams, extrinsic_convention='world2cam', assume_no_distortion=True):
    """
    cams: dict with keys 'cam1', 'cam2' each containing intrinsic, extrinsic, resolution
    extrinsic_convention: 'world2cam' or 'cam2world' (choose the one that matches your data)
    assume_no_distortion: if True sets dist vectors to zeros (length 5)
    Returns:
      mtx_l, dist_l, mtx_r, dist_r, R_rel, T_rel
    where R_rel, T_rel satisfy: X_r = R_rel @ X_l + T_rel
    """
    K1, ext1, res1 = parse_camera_dict(cams['cam1'])
    K2, ext2, res2 = parse_camera_dict(cams['cam2'])

    # intrinsics
    mtx_l = K1.copy()
    mtx_r = K2.copy()

    # distortion (not provided in input) -> zeros (k1,k2,p1,p2,k3)
    if assume_no_distortion:
        dist_l = np.zeros((5,), dtype=float)
        dist_r = np.zeros((5,), dtype=float)
    else:
        dist_l = np.zeros((5,), dtype=float)
        dist_r = np.zeros((5,), dtype=float)

    # convert extrinsics -> world->camera convention
    R1_wc, t1_wc = extrinsic_to_Rt(ext1, convention=extrinsic_convention)
    R2_wc, t2_wc = extrinsic_to_Rt(ext2, convention=extrinsic_convention)

    # compute relative transform from cam1 to cam2:
    # X_r = R_rel * X_l + T_rel
    R_rel = R2_wc @ R1_wc.T
    T_rel = t2_wc - R_rel @ t1_wc  # shape (3,1)

    return mtx_l, dist_l, mtx_r, dist_r, R_rel, T_rel

def skew_from_vec(t):
    """给定 (3,) 或 (3,1) 向量 t，返回 3x3 的叉乘矩阵 [t]_x"""
    t = np.asarray(t).reshape(3,)
    tx, ty, tz = t
    return np.array([[0, -tz, ty],
                     [tz, 0, -tx],
                     [-ty, tx, 0]], dtype=float)

def compute_E_F_P(P_mtx_left, P_mtx_right, R, T):
    """
    输入:
      P_mtx_left: 3x3 numpy (K_left)
      P_mtx_right: 3x3 numpy (K_right)
      R: 3x3 rotation (from left->right: X_r = R X_l + T)
      T: 3x1 or (3,) translation (from left->right)
    返回:
      E, F, P1, P2
      - E: 3x3 essential matrix
      - F: 3x3 fundamental matrix (pixel coords)
      - P1: 3x4 projection matrix for left (K [I|0])
      - P2: 3x4 projection matrix for right (K [R|T])
    """
    K1 = np.asarray(P_mtx_left, dtype=float)
    K2 = np.asarray(P_mtx_right, dtype=float)
    R = np.asarray(R, dtype=float)
    T = np.asarray(T, dtype=float).reshape(3,)

    # 1) E
    Tx = skew_from_vec(T)
    E = Tx.dot(R)

    # 2) F = K2^{-T} E K1^{-1}
    F = np.linalg.inv(K2).T.dot(E).dot(np.linalg.inv(K1))

    # Optional: enforce rank-2 on E and F (numerical)
    # enforce rank-2 by SVD for E and F if desired:
    # Ue, Se, Vte = np.linalg.svd(E); Se[2]=0; E_rank2 = Ue @ np.diag(Se) @ Vte
    # Uf, Sf, Vtf = np.linalg.svd(F); Sf[2]=0; F_rank2 = Uf @ np.diag(Sf) @ Vtf

    # 3) P1, P2
    P1 = K1.dot(np.hstack((np.eye(3), np.zeros((3,1)))))
    P2 = K2.dot(np.hstack((R, T.reshape(3,1))))

    return E, F, P1, P2

def json_to_mask(json_path, image_shape):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    H, W = image_shape
    mask = np.zeros((H, W), dtype=np.uint8)
    shape = data["shapes"][0]
    start_point = shape["points"][0]
    end_point = shape["points"][-1]
    points = np.array(shape["points"], dtype=np.int32)
    for i in range(len(points) - 1):
        pt1 = tuple(points[i])
        pt2 = tuple(points[i+1])
        cv2.line(mask, pt1, pt2, color=1, thickness=1)  # thickness 可调导丝粗细
    return mask, start_point, end_point

def find_intersections(mask, line, tol=1.0):
    a, b, c = line
    ys, xs = np.where(mask == 1)
    dist = np.abs(a*xs + b*ys + c) / np.sqrt(a*a + b*b)
    idx = np.where(dist < tol)[0]
    return list(zip(xs[idx], ys[idx]))

def draw_epiline(img, line):
    a, b, c = line
    h, w = img.shape[:2]
    x0, y0 = 0, int(-c / b) if b != 0 else 0
    x1, y1 = w, int(-(c + a * w) / b) if b != 0 else h
    return cv2.line(img, (x0,y0), (x1,y1), 1, 1)


def get_camera_center(P):
    U, S, Vt = np.linalg.svd(P)
    C_h = Vt[-1]
    C_h /= C_h[-1]
    return C_h[0:3]

def reconstruct_polyline_3d(polyline_left, mask2, P1, P2, F):
    pts_3d = []
    if(len(np.where(mask2==1)[0])==0):
        return np.array(pts_3d)

    polyline_right = ordered_points_from_mask(mask2)
    pre_2d_point = (-1,-1)
    tol = 1.0

    idx_prev_right = -1
    uv_prev_right = (-1,-1)

    cnt = 0

    for (v_l, u_l) in polyline_left:
        line_right = cv2.computeCorrespondEpilines(np.array([[[u_l, v_l]]], dtype=np.float32), 1, F).reshape(-1, 3)[0]
        a, b, c = line_right
        # img = mask2
        # img = draw_epiline(img, line_right)
        # cv2.imwrite(f"epiline_{u_l}_{v_l}.png", img*255)
        # quit()
        ys, xs = polyline_right[:,0], polyline_right[:,1]
        dist = a*xs + b*ys + c / np.sqrt(a*a + b*b)
        pts_dist = list(zip(xs, ys, dist))

        idx_list = []
        for i in range(len(pts_dist)-1):
            if(pts_dist[i][2]*pts_dist[i+1][2]<0):
                idx_list.append(i)
        if(len(idx_list)==1):
            idx_prev_right = idx_list[0]
            uv_prev_right = (pts_dist[idx_prev_right][1], pts_dist[idx_prev_right][0])
        elif(len(idx_list)>1):
            uv_dists = [np.sqrt((pts_dist[idx][1]-u_l)**2+(pts_dist[idx][0]-v_l)**2) for idx in idx_list]
            min_idx = np.argmin(uv_dists)
            idx_prev_right = idx_list[min_idx]
            uv_prev_right = (pts_dist[idx_prev_right][1], pts_dist[idx_prev_right][0])
        
        pt_l = np.array([u_l, v_l], dtype=np.float32).reshape(2, 1)
        pt_r = np.array(uv_prev_right, dtype=np.float32).reshape(2, 1)


        # sum_dist = pt1[2] + pt2[2]
        # x_weighted = pt1[0] * (pt2[2] / sum_dist) + pt2[0] * (pt1[2] / sum_dist)
        # y_weighted = pt1[1] * (pt2[2] / sum_dist) + pt2[1] * (pt1[2] / sum_dist)
        # pt_l = np.array([u_l, v_l], dtype=np.float32).reshape(2, 1)
        # pt_r = np.array([x_weighted, y_weighted], dtype=np.float32).reshape(2, 1)
        # pre_2d_point = (x_weighted, y_weighted)
        # print(pre_2d_point)

        pt_l_ud = cv2.undistortPoints(np.expand_dims(pt_l, axis=1), mtx_l, dist_l)
        pt_r_ud = cv2.undistortPoints(np.expand_dims(pt_r, axis=1), mtx_r, dist_r)
        points4D = cv2.triangulatePoints(P1, P2, pt_l_ud.T, pt_r_ud.T)
        pt_3d = (points4D[:3] / points4D[3]).flatten()  # (x,y,z)
        # print(pt_3d)
        # pts_4d, err = triangulate_closest(P1, P2, pt_l, pt_r)
        # pt_3d = (pts_4d[:3] / pts_4d[3]).ravel()
        pts_3d.append(pt_3d)
        # print(pt_3d)
    return np.array(pts_3d)

def project_with_P(points_3d, P, image_size, intensity=255):
    pts = np.asarray(points_3d, dtype=np.float64)
    pts_h = np.hstack([pts, np.ones((pts.shape[0],1))])
    uvw = (P @ pts_h.T).T  # (N,3)
    uv = uvw[:, :2] / uvw[:, 2:3]

    h, w = image_size
    img = np.zeros((h, w), dtype=np.uint8)
    for (u,v) in uv:
        u = int(round(u))
        v = int(round(v))
        if 0 <= u < w and 0 <= v < h:
            img[v, u] = intensity
    return img, uv

# REAL CAMERA
# params = np.load("real_data/calibration/stereo_params.npz")
# mtx_l, dist_l = params["mtx_l"], params["dist_l"]
# mtx_r, dist_r = params["mtx_r"], params["dist_r"]
# R, T = params["R"], params["T"]
# F = params["F"]
# P1 = mtx_l @ np.hstack((np.eye(3), np.zeros((3,1))))
# P2 = mtx_r @ np.hstack((R, T))
# mask1, start_point1, end_point1 = json_to_mask("real_data/mask/C1.json", (3000, 4096))
# mask2, start_point2, end_point2 = json_to_mask("real_data/mask/C2.json", (3000, 4096))

# SIM CAMERA
para = json.load(open('sim_data/calibration/camera_params.json'))
mtx_l, dist_l, mtx_r, dist_r, R, T = get_stereo_params_from_dict(para, extrinsic_convention='world2cam')
E, F, P1, P2 = compute_E_F_P(mtx_l, mtx_r, R, T)
mask1 = (cv2.imread("sim_data/mask/mask_1.png", cv2.IMREAD_GRAYSCALE)/ 255).astype(np.uint8)
mask2 = (cv2.imread("sim_data/mask/mask_2.png", cv2.IMREAD_GRAYSCALE)/ 255).astype(np.uint8)

polyline_left = ordered_points_from_mask(mask1)
# polyline_right = ordered_points_from_mask(mask2)

curve_3d = reconstruct_polyline_3d(polyline_left, mask2, P1, P2, F)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(curve_3d)
o3d.io.write_point_cloud("sim_data/reconstruction/guidewire.ply", pcd)

# img_left, uv_left = project_with_P(curve_3d, P1, image_size=(512, 2048))
# img_right, uv_right = project_with_P(curve_3d, P2, image_size=(512, 2048))
# cv2.imwrite("proj_left.png", img_left)
# cv2.imwrite("proj_right.png", img_right)