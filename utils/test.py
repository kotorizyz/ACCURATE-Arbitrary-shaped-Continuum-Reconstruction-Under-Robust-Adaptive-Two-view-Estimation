import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize_3d

import argparse

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

def compute_point_to_curve_metrics_simulation(pred, gt):
    d_min, _ = point_to_polyline_distance(pred, gt)
    rmse = np.sqrt(np.mean(d_min**2))
    mae = np.mean(d_min)
    max_err = np.max(d_min)
    return mae, max_err

def compute_point_to_curve_metrics_phantom(pred, gt):
    # pred [N', 3], gt [N, 3]
    tree = cKDTree(pred)
    dist, _ = tree.query(gt, k=1)

    mae = dist.mean()
    max_err = dist.max()
    return mae, max_err

def compute_acc_comp(
    gt: np.ndarray,
    pred: np.ndarray,
    tau: float = None,
):

    gt = np.asarray(gt, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)

    assert gt.ndim == 2 and gt.shape[1] == 3
    assert pred.ndim == 2 and pred.shape[1] == 3

    # Build KD-Trees
    tree_gt = cKDTree(gt)
    tree_pred = cKDTree(pred)

    # Accuracy: pred -> gt
    dist_acc, _ = tree_gt.query(pred, k=1)
    acc = float(dist_acc.mean())

    # Completeness: gt -> pred
    dist_comp, _ = tree_pred.query(gt, k=1)
    comp = float(dist_comp.mean())

    return acc, comp

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--category',
        type=str,
        default='phantom',
        help='Dataset category: simulation or phantom (default: phantom)'
    )

    parser.add_argument(
        '--receive',
        type=str,
        default='mask',
        help='Input type: image, mask or point (default: mask)'
    )

    parser.add_argument(
        '--method',
        type=str,
        default='ACCURATE',
        help='Reconstruction method: ACCURATE/TMI03/TMI15/vggt/Fast3R/MonSter (default: ACCURATE)'
    )
    args = parser.parse_args()

    category = args.category
    receive = args.receive
    method = args.method

    test_names = []
    with open(f'ACCURATE_dataset/splits/{category}_test.txt', 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            test_names.append(line.strip())
            line = f.readline()

    sum_mae = 0
    sum_max_err = 0
    sum_acc = 0
    sum_comp = 0
    for test_name in test_names:
        if category == 'phantom' and int(test_name[-3:]) < 50: continue # There is something wrong with the GT of phantom 1-49, we didn't use the result of case 1 in paper for all methods. Skip them for now.
        gt = o3d.io.read_point_cloud(f'ACCURATE_dataset/{category}/{test_name}/annotations/guidewire_3D.ply')
        gt = np.asarray(gt.points)

        if receive == 'image':
            rec = o3d.io.read_point_cloud(f'experiment/{receive}/{category}/prediction_results/{test_name}.ply')
        elif receive == 'mask':
            rec = o3d.io.read_point_cloud(f'experiment/{receive}/{category}/{method}/{test_name}.ply')
        elif receive == 'point':
            rec = o3d.io.read_point_cloud(f'experiment/{receive}/{category}/{method}/{test_name}.ply')

        rec = np.asarray(rec.points)

        # if category == 'phantom':
        #     mae, max_err = compute_point_to_curve_metrics_phantom(rec, gt)
        # else:
        mae, max_err = compute_point_to_curve_metrics_phantom(rec, gt)
        
        acc, comp = compute_acc_comp(gt, rec)
        # acc = result['acc']
        # comp = result['comp']
        sum_mae += mae
        sum_max_err += max_err
        sum_acc += acc
        sum_comp += comp
        # print(test_name, acc, comp, mae, max_err)
    print(f'Average MAE: {sum_mae/len(test_names)}')
    print(f'Average Accuracy: {sum_acc/len(test_names)}')
    print(f'Average Completeness: {sum_comp/len(test_names)}')
    print(f'Overall: {(sum_acc + sum_comp) / len(test_names) / 2}')
    print(f'Average Max Error: {sum_max_err/len(test_names)}')