import numpy as np
import cv2
import os
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

def dilate_mask(mask, radius):
    mask_u8 = (mask > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * radius + 1, 2 * radius + 1)
    )
    dilated = cv2.dilate(mask_u8, kernel)
    return dilated.astype(bool)

def extract_object_points_dual_view_relaxed(
    points_3d,
    P1, P2,
    mask1, mask2,
    image_shape,
    dilation_radius=3
):
    H, W = image_shape

    mask1_d = dilate_mask(mask1, dilation_radius)
    mask2_d = dilate_mask(mask2, dilation_radius)

    N = points_3d.shape[0]
    valid_mask = np.zeros(N, dtype=bool)

    points_h = np.hstack([points_3d, np.ones((N, 1))])

    img_L = np.zeros((H, W), dtype=np.uint8)
    projected_L = (P1 @ points_h.T).T
    projected_L = projected_L[:, :2] / projected_L[:, 2:3]

    for i in range(N):
        X = points_h[i]

        keep = True
        for P, mask in [(P1, mask1_d), (P2, mask2_d)]:
            proj = P @ X

            if proj[2] <= 0:
                keep = False
                break

            u = proj[0] / proj[2]
            v = proj[1] / proj[2]

            ui, vi = int(round(u)), int(round(v))

            if ui < 0 or ui >= W or vi < 0 or vi >= H:
                keep = False
                break

            if not mask[vi, ui]:
                keep = False
                break

        if keep:
            valid_mask[i] = True

    return points_3d[valid_mask]

def pointcloud_to_depth_KRT(
    pts3d,
    K, R, t,
    H, W
):
    depth = np.ones((H, W)) * 10000

    pts3d = pts3d.astype(np.float64)
    R = R.astype(np.float64)
    t = t.reshape(3, 1).astype(np.float64)

    # ---- world -> camera ----
    Pc = (R @ pts3d.T + t).T
    Xc, Yc, Zc = Pc[:, 0], Pc[:, 1], Pc[:, 2]

    # ---- camera -> pixel ----
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = (fx * Xc / Zc + cx).astype(np.int32)
    v = (fy * Yc / Zc + cy).astype(np.int32)

    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, Zc = u[inside], v[inside], Zc[inside]

    # ---- Z-buffer (keep nearest point) ----
    for ui, vi, zi in zip(u, v, Zc):
        if zi < depth[vi, ui]:
            depth[vi, ui] = zi
    return depth

def reconstruct_points_optimize_depth_scale_camera(
    depth,          # (H,W)
    mask,           # (H,W)
    K,              # (3,3)
    R,              # (3,3)
    t,              # (3,)
    pts_3d_gt,      # (M,3)
    dilation_radius=2,
    iters=3
):
    """
    在相机坐标系下优化深度尺度 alpha，
    再变换到世界坐标系，与 GT 对齐
    """

    # ---------- 1. mask 膨胀 ----------
    if dilation_radius > 0:
        kernel = np.ones((2*dilation_radius+1, 2*dilation_radius+1), np.uint8)
        mask_use = cv2.dilate(mask.astype(np.uint8), kernel) > 0
    else:
        mask_use = mask > 0

    ys, xs = np.where(mask_use)
    Z = depth[ys, xs]

    valid = Z > 0
    xs, ys, Z = xs[valid], ys[valid], Z[valid]

    # ---------- 2. 像素 → 相机坐标（alpha = 1） ----------
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    X = (xs - cx) * Z / fx
    Y = (ys - cy) * Z / fy

    pts_cam = np.stack([X, Y, Z], axis=1)  # (N,3)

    # 预计算 R * x_c
    A = (R @ pts_cam.T).T   # (N,3)

    tree = cKDTree(pts_3d_gt)
    alpha = 1.0

    for _ in range(iters):
        pts_world = alpha * A + t
        _, idx = tree.query(pts_world, k=1)

        Ygt = pts_3d_gt[idx]

        num = np.sum(np.sum(A * (Ygt - t), axis=1))
        den = np.sum(np.sum(A * A, axis=1)) + 1e-8

        alpha = num / den

    pts_world_final = alpha * A + t
    return pts_world_final, alpha

def optimal_scale_l1(gt, pred, eps=1e-8):
    """
    Solve min_alpha sum |gt - alpha * pred|  (L1 / MAE)

    Args:
        gt, pred: (N,) arrays
        eps: avoid division by zero

    Returns:
        alpha: optimal scale factor
    """

    gt = np.asarray(gt)
    pred = np.asarray(pred)

    valid = np.abs(pred) > eps
    ratios = gt[valid] / pred[valid]

    alpha = np.median(ratios)
    return alpha


def depth_mask_to_world(
    depth_eff_pred,   # (N,)
    maskL,             # (H, W), mask == 1
    K,                 # (3, 3)
    R, t               # world -> camera
):
    """
    Back-project masked depth values to world coordinates.

    Returns:
        pts_world: (N, 3)
    """

    # ---------- 1. mask -> pixel coordinates ----------
    v, u = np.where(maskL == 1)   # v: row (y), u: col (x)

    assert len(depth_eff_pred) == len(u)

    Z = depth_eff_pred.astype(np.float64)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # ---------- 2. pixel + depth -> camera coordinates ----------
    Xc = (u - cx) * Z / fx
    Yc = (v - cy) * Z / fy
    Zc = Z

    pts_cam = np.stack([Xc, Yc, Zc], axis=1)   # (N, 3)

    # ---------- 3. camera -> world ----------
    R = R.astype(np.float64)
    t = t.reshape(3,).astype(np.float64)

    pts_world = (R.T @ (pts_cam - t).T).T

    return pts_world


if __name__ == '__main__':
    category = 'phantom'
    method = 'MonSter'

    data_path = f'ACCURATE_dataset/{category}'
    test_names = []
    with open(f'ACCURATE_dataset/splits/{category}_test.txt', 'r', encoding='utf-8') as f:
        fileline = f.readline()
        while fileline:
            test_names.append(fileline.strip())
            fileline = f.readline()
    
    for test_name in test_names:
        test_path = os.path.join(data_path, test_name)
        maskL_path = os.path.join(test_path, 'masks', 'mask_L.png')
        maskR_path = os.path.join(test_path, 'masks', 'mask_R.png')
        maskL = cv2.imread(maskL_path, cv2.IMREAD_GRAYSCALE) > 0
        maskR = cv2.imread(maskR_path, cv2.IMREAD_GRAYSCALE) > 0

        K_L = np.loadtxt(os.path.join(test_path, 'calibration', 'K_L.txt'))
        RT_L = np.loadtxt(os.path.join(test_path, 'calibration', 'RT_L.txt'))
        P_L = K_L @ RT_L

        K_R = np.loadtxt(os.path.join(test_path, 'calibration', 'K_R.txt'))
        RT_R = np.loadtxt(os.path.join(test_path, 'calibration', 'RT_R.txt'))
        P_R = K_R @ RT_R

        img_shape = maskL.shape
        if method == 'Fast3R':
            depthL_path = f'experiment/{category}/{method}/{test_name}/image/view_L.npy'
            depthR_path = f'experiment/{category}/{method}/{test_name}/image/view_R.npy'
            depth_L = np.load(depthL_path)
            depth_R = np.load(depthR_path)
            if category == 'simulation':
                depth_L = cv2.resize(depth_L, (512, 2048))
                depth_R = cv2.resize(depth_R, (512, 2048))
            elif category == 'phantom':
                depth_L = cv2.resize(depth_L, (500, 500))
                depth_R = cv2.resize(depth_R, (500, 500))
        else:
            depthL_path = f'experiment/{category}/{method}/{test_name}_L.npy'
            depthR_path = f'experiment/{category}/{method}/{test_name}_R.npy'
            depth_L = np.load(depthL_path)
            depth_R = np.load(depthR_path)

        # pts_3d_R = depth_mask_to_world_points(depth_R_500, maskR, K_R, RT_R[:,:3], RT_R[:,3], dilation_radius=2)
        # o3d.io.write_point_cloud(f'test.ply', o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_3d_L)))

        pts_3d_gt = o3d.io.read_point_cloud(f'ACCURATE_dataset/{category}/{test_name}/annotations/guidewire_3D.ply')
        pts_3d_gt = np.asarray(pts_3d_gt.points)

        depth_L_gt = pointcloud_to_depth_KRT(
            pts_3d_gt,
            K_L, RT_L[:,:3], RT_L[:,3],
            img_shape[0], img_shape[1]
        )
        maskL = depth_L_gt != 10000
        depth_L_eff_gt = depth_L_gt[maskL]
        depth_L_eff_pred = depth_L[maskL]
        alpha_L = optimal_scale_l1(depth_L_eff_gt, depth_L_eff_pred)
        # pred_scaled = alpha * depth_L_mask
        pts_3d_L_pred = depth_mask_to_world(
            alpha_L * depth_L_eff_pred,
            maskL,
            K_L,
            RT_L[:,:3],
            RT_L[:,3]
        )

        depth_R_gt = pointcloud_to_depth_KRT(
            pts_3d_gt,
            K_R, RT_R[:,:3], RT_R[:,3],
            img_shape[0], img_shape[1]
        )
        maskR = depth_R_gt != 10000
        depth_R_eff_gt = depth_R_gt[maskR]
        depth_R_eff_pred = depth_R[maskR]
        alpha_R = optimal_scale_l1(depth_R_eff_gt, depth_R_eff_pred)
        # pred_scaled = alpha * depth_L_mask
        pts_3d_R_pred = depth_mask_to_world(
            alpha_R * depth_R_eff_pred,
            maskR,
            K_R,
            RT_R[:,:3],
            RT_R[:,3]
        )
        print(alpha_L, alpha_R)
        
        pts_3d_pred = np.vstack([pts_3d_L_pred, pts_3d_R_pred])
        o3d.io.write_point_cloud(f'experiment/{category}/{method}/{test_name}.ply', o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_3d_pred)))