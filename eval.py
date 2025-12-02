import numpy as np
import open3d as o3d

def point_to_segment_distances(points, seg_start, seg_end):
    P = points[:, None, :]          # (M,1,3)
    A = seg_start[None, :, :]       # (1,S,3)
    B = seg_end[None, :, :]         # (1,S,3)
    AB = B - A                      # (1,S,3)
    AP = P - A                      # (M,S,3)

    ab2 = np.sum(AB * AB, axis=2)           # (1,S)
    ab2_safe = np.where(ab2 == 0, 1e-12, ab2)

    t = np.sum(AP * AB, axis=2) / ab2_safe   # (M,S)

    t_clip = np.clip(t, 0.0, 1.0)            # (M,S)

    Q = A + t_clip[..., None] * AB           # (M,S,3)

    diff = P - Q                             # (M,S,3)
    dists = np.linalg.norm(diff, axis=2)     # (M,S)
    return dists

def point_to_polyline_distance(points, polyline):
    if polyline.shape[0] < 2:
        d_min = np.linalg.norm(points - polyline[0][None,:], axis=1)
        return d_min, np.zeros(len(points), dtype=int)

    A = polyline[:-1]
    B = polyline[1:]
    dists = point_to_segment_distances(points, A, B)  # (M, S)
    argmin_seg = np.argmin(dists, axis=1)
    d_min = dists[np.arange(dists.shape[0]), argmin_seg]
    return d_min, argmin_seg

def compute_point_to_curve_metrics(pred, gt):
    d_min, _ = point_to_polyline_distance(pred, gt)
    rmse = np.sqrt(np.mean(d_min**2))
    mae = np.mean(d_min)
    max_err = np.max(d_min)
    return d_min, rmse, mae, max_err

def chamfer_distance(pred, gt):
    d_pred, _ = point_to_polyline_distance(pred, gt)   # (N,)
    d_gt, _ = point_to_polyline_distance(gt, pred)     # (N0,)
    chamfer = np.mean(d_pred) + np.mean(d_gt)
    hausdorff = max(np.max(d_pred), np.max(d_gt))
    return chamfer, hausdorff, d_pred, d_gt

for i in range(10):
    gt = o3d.io.read_point_cloud(f'output/gt_{i}.ply')
    gt = np.asarray(gt.points)
    pred = o3d.io.read_point_cloud(f'output/rec_{i}.ply')
    pred = np.asarray(pred.points)
    # print(gt.shape, pred.shape)
    d_pred, rmse, mae, max_err = compute_point_to_curve_metrics(pred, gt)
    print(f"{i:<3d}  RMSE: {rmse:<10.6f}  MAE: {mae:<10.6f}  Max: {max_err:<10.6f}")