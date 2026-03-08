import numpy as np
import cv2
from skimage.morphology import skeletonize
from skimage.draw import line
from scipy import linalg
import torch
import open3d as o3d
import os

import argparse

def compute_stereo_params(K1,K2,R1,R2,t1,t2):
    R = R2 @ R1.T
    T = t2 - R @ t1
    
    def skew(v):
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
    E = skew(T) @ R
    
    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)

    P1 = K1 @ np.concatenate((R1, t1[:,None]), axis=1)
    P2 = K2 @ np.concatenate((R2, t2[:,None]), axis=1)
    return R, T, E, F, P1, P2

def fix_gaps(mask, max_gap=1):
    skel = skeletonize(mask>0).astype(np.uint8)
    points = np.argwhere(skel>0)

    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skel, -1, kernel)
    deg_map = (skel == 1) & (neighbor_count == 1)
    end_points = [tuple(p) for p in points if deg_map[p[0],p[1]]]

    for i, p1 in enumerate(end_points):
        for j, p2 in enumerate(end_points):
            if i >= j:
                continue
            dist = max(abs(p1[0]-p2[0]), abs(p1[1]-p2[1]))
            if 1 <= dist <= max_gap:
                rr, cc = line(p1[0], p1[1], p2[0], p2[1])
                skel[rr, cc] = 1

    neighbor_count = cv2.filter2D(skel, -1, kernel)
    deg_map = (skel == 1) & (neighbor_count == 1)
    end_points = [tuple(p) for p in points if deg_map[p[0],p[1]]]

    return skel, end_points

def calculate_epi_dist(points_L, points_R, F):
    N1 = points_L.shape[0]
    N2 = points_R.shape[0]
    dist = np.zeros((N1, N2))
    line_right = cv2.computeCorrespondEpilines(points_L[:,::-1], 1, F).reshape(-1, 3)

    for i in range(N1):
            a, b, c = line_right[i]
            xs = points_R[:,1]
            ys = points_R[:,0]
            num = a*xs + b*ys + c
            denom = np.sqrt(a*a + b*b)
            dist[i] = num / denom
    
    return dist

def get_candidates(mask, curr, visited, r_min=1, r_max=2):
    H, W = mask.shape
    candidates = []
    for r in range(r_min, r_max+1):
        for dx in range(-r, r+1):
            for dy in range(-r, r+1):
                nx, ny = curr[0]+dx, curr[1]+dy
                if 0 <= nx < H and 0 <= ny < W and mask[nx, ny]>0 and (nx, ny) not in visited:
                    dist = max(abs(dx), abs(dy))
                    candidates.append(((nx, ny), dist))
        if(len(candidates) > 0): break
    return candidates

def fit_curve(points):
    pts = np.array(points)
    if len(points)<3:
        return np.zeros(3), np.zeros(3), 1e10, 1e10
    t = np.linspace(0,1,len(points))
    A = np.vstack([t**2, t, np.ones_like(t)]).T
    coef_x, res_x, _, _ = linalg.lstsq(A, pts[:,0])
    coef_y, res_y, _, _ = linalg.lstsq(A, pts[:,1])
    return coef_x, coef_y, res_x, res_y

def curvature_at_t(coef_x, coef_y, t):
    a_x, b_x, _ = coef_x
    a_y, b_y, _ = coef_y
    x_prime = 2*a_x*t + b_x
    x_double = 2*a_x
    y_prime = 2*a_y*t + b_y
    y_double = 2*a_y
    k = abs(x_prime*y_double - y_prime*x_double) / ((x_prime**2 + y_prime**2)**1.5 + 1e-8)
    return k

def traverse_curve(mask, window_size=10, r_min=1, r_max=2, gap_threshold=3, start_point=None, lmda = 1.0, lmdd = 0.5):
    visited = set()
    sequence = [start_point]
    visited.add(start_point)
    prev_point = None
    curr_point = start_point

    eps = 1e-4
    while True:
        candidates = get_candidates(mask, curr_point, visited, r_min=r_min, r_max=r_max)
        if len(candidates)==0:
            break
        elif len(candidates)==1:
            next_point = candidates[0][0]
        else:
            window = sequence[-window_size:]
            min_score = float('inf')
            min_res = float('inf')
            next_point = candidates[0][0]
            for p, dist in candidates:
                pts = window + [p]
                coef_x, coef_y, res_x, res_y = fit_curve(pts)
                k = curvature_at_t(coef_x, coef_y, t=1.0)
                if prev_point is not None:
                    v_prev = np.array(curr_point)-np.array(prev_point)
                    u = np.array(p)-np.array(curr_point)
                    angle_penalty = 1 - np.dot(v_prev,u)/(np.linalg.norm(v_prev)*np.linalg.norm(u)+1e-8)
                else:
                    angle_penalty = 0
                score = k + lmda * angle_penalty + lmdd * dist
                if score < min_score:
                    if abs(score - min_score) < eps:
                        if res_x + res_y < min_res:
                            min_res = res_x + res_y
                            min_score = score
                            next_point = p
                    else:
                        min_res = res_x + res_y
                        min_score = score
                        next_point = p
        sequence.append(next_point)
        visited.add(next_point)
        prev_point = curr_point
        curr_point = next_point
    return sequence

def gctt(mask_L, mask_R, F):
    # fix gaps & get endpoints
    mask_L, end_points_L = fix_gaps(mask_L)
    mask_R, end_points_R = fix_gaps(mask_R)

    # calculate endpoints epi distance
    dist_endpoints = np.abs(calculate_epi_dist(np.array(end_points_L), np.array(end_points_R), F))
    start_point_L = end_points_L[np.unravel_index(np.argmin(dist_endpoints), dist_endpoints.shape)[0]]
    start_point_R = end_points_R[np.unravel_index(np.argmin(dist_endpoints), dist_endpoints.shape)[1]]

    # traversal
    seq_L = traverse_curve(mask_L, start_point=start_point_L, r_min=10, r_max=50)
    seq_R = traverse_curve(mask_R, start_point=start_point_R, r_min=10, r_max=50)

    return seq_L, seq_R

def ecdp(seq_L, seq_R, F, refine):
    D = np.abs(calculate_epi_dist(np.array(seq_L), np.array(seq_R), F))
    N1, N2 = D.shape
    C = np.zeros_like(D, dtype=float)
    parent = np.full((N1, N2, 2), -1, dtype=int)

    C[0,0] = D[0,0]
    for j in range(1, N2):
        C[0,j] = C[0,j-1] + D[0,j]
        parent[0,j] = [0, j-1]
    for i in range(1, N1):
        C[i,0] = C[i-1,0] + D[i,0]
        parent[i,0] = [i-1, 0]
    for i in range(1, N1):
        for j in range(1, N2):
            if C[i-1,j] == min(C[i-1,j], C[i,j-1], C[i-1,j-1]):
                C[i,j] = C[i-1,j] + D[i,j]
                parent[i,j] = [i-1, j]
            elif C[i,j-1] == min(C[i-1,j], C[i,j-1], C[i-1,j-1]):
                C[i,j] = C[i,j-1] + D[i,j]
                parent[i,j] = [i, j-1]
            else:
                C[i,j] = C[i-1,j-1] + D[i,j]
                parent[i,j] = [i-1, j-1]

    path = []
    i, j = N1-1, N2-1
    while i != -1 and j != -1:
        path.append((i,j))
        pi, pj = parent[i,j]
        i, j = pi, pj

    path.reverse()
    return path

def reconstruction(seq_L, seq_R, matches, P1, P2):
    pts_3d = []
    for match in matches:
        pt_L = np.array(seq_L[match[0]][::-1], dtype=np.float32).reshape(2, 1)
        pt_R = np.array(seq_R[match[1]][::-1], dtype=np.float32).reshape(2, 1)
        points4D = cv2.triangulatePoints(P1, P2, pt_L, pt_R)
        pt_3d = (points4D[:3] / points4D[3]).flatten()
        pts_3d.append(pt_3d)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_3d)
    return pcd

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Wire reconstruction experiment settings")

    parser.add_argument(
        '--category',
        type=str,
        default='phantom',
        help='Dataset category: simulation or phantom (default: simulation)'
    )

    parser.add_argument(
        '--receive',
        type=str,
        default='mask',
        help='Input type: image or mask (default: mask)'
    )

    parser.add_argument(
        '--method',
        type=str,
        default='ACCURATE',
        help='Reconstruction method (default: ACCURATE)'
    )

    parser.add_argument(
        '--refine',
        action='store_true',
        help='Enable refinement step (default: False)'
    )

    args = parser.parse_args()

    category = args.category
    receive = args.receive
    method = args.method
    refine = args.refine

    if receive == 'image':
        data_path = f'experiment/{receive}/{category}/prediction_results'
    elif receive == 'mask':
        data_path = f'ACCURATE_dataset/{category}'
    
    test_names = []
    with open(f'ACCURATE_dataset/splits/{category}_test.txt', 'r', encoding='utf-8') as f:
        fileline = f.readline()
        while fileline:
            test_names.append(fileline.strip())
            fileline = f.readline()

    for test_name in test_names:
        # print(test_name)
        if receive == 'image':
            mask_L = cv2.imread(os.path.join(data_path, f'{test_name}_L.png'), cv2.IMREAD_UNCHANGED)
            mask_R = cv2.imread(os.path.join(data_path, f'{test_name}_R.png'), cv2.IMREAD_UNCHANGED)
        elif receive == 'mask':
            mask_L = cv2.imread(os.path.join(data_path, test_name, 'masks/mask_L.png'), cv2.IMREAD_UNCHANGED)
            mask_R = cv2.imread(os.path.join(data_path, test_name, 'masks/mask_R.png'), cv2.IMREAD_UNCHANGED)
        
        if category == 'phantom':
            mask_L = skeletonize(mask_L>0).astype(np.uint8)
            mask_R = skeletonize(mask_R>0).astype(np.uint8)

        # camera params
        K_L = np.loadtxt(f'ACCURATE_dataset/{category}/{test_name}/calibration/K_L.txt')
        K_R = np.loadtxt(os.path.join(f'ACCURATE_dataset/{category}/{test_name}/calibration/K_R.txt'))
        RT_L = np.loadtxt(os.path.join(f'ACCURATE_dataset/{category}/{test_name}/calibration/RT_L.txt'))
        RT_R = np.loadtxt(os.path.join(f'ACCURATE_dataset/{category}/{test_name}/calibration/RT_R.txt'))
        R_L = RT_L[:,:3]
        t_L = RT_L[:,3]
        R_R = RT_R[:,:3]
        t_R = RT_R[:,3]
        R, T, E, F, P1, P2 = compute_stereo_params(K_L,K_R,R_L,R_R,t_L,t_R)

        seq_L, seq_R = gctt(mask_L, mask_R, F)
        matches = ecdp(seq_L, seq_R, F)

        pcd = reconstruction(seq_L, seq_R, matches, P1, P2)
        o3d.io.write_point_cloud(f"experiment/{receive}/{category}/{method}/{test_name}.ply", pcd)