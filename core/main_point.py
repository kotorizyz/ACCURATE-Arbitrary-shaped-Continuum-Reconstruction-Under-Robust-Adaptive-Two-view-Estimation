import numpy as np
import cv2
from skimage.morphology import skeletonize
from skimage.draw import line
from scipy import linalg
import torch
import open3d as o3d
import os
import matplotlib.pyplot as plt
import argparse

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

def project_points(pts_3d, K, R, t):

    pts_cam = (R @ pts_3d.T + t.reshape(3,1))  # [3,N]

    pts_img = K @ pts_cam  # [3,N]

    u = pts_img[0] / pts_img[2]
    v = pts_img[1] / pts_img[2]

    return np.stack([u, v], axis=1)  # [N,2]

def project_with_depth(pts_3d, K, R, t):
    pts_cam = R @ pts_3d.T + t.reshape(3,1)   # [3,N]

    z = pts_cam[2]

    pts_img = K @ pts_cam
    u = pts_img[0] / z
    v = pts_img[1] / z

    return np.stack([u, v, z], axis=1)        # [N,3]

def visible_order(pts_uvz, H, W):

    z_buffer = np.full((H, W), np.inf)
    idx_buffer = -np.ones((H, W), dtype=int)

    for i, (u, v, z) in enumerate(pts_uvz):

        u_pix = int(round(u))
        v_pix = int(round(v))

        if not (0 <= u_pix < W and 0 <= v_pix < H):
            continue

        if z < z_buffer[v_pix, u_pix]:
            z_buffer[v_pix, u_pix] = z
            idx_buffer[v_pix, u_pix] = i

    visible_indices = np.unique(idx_buffer[idx_buffer >= 0])
    visible_indices.sort()

    return visible_indices

def ordered_visible_pixels(pts_uvz, H, W):
    z_buffer = np.full((H, W), np.inf)
    idx_buffer = -np.ones((H, W), dtype=int)

    for i, (u, v, z) in enumerate(pts_uvz):

        u_pix = int(round(u))
        v_pix = int(round(v))

        if not (0 <= u_pix < W and 0 <= v_pix < H):
            continue
        if z < z_buffer[v_pix, u_pix]:
            z_buffer[v_pix, u_pix] = z
            idx_buffer[v_pix, u_pix] = i

    visible_indices = idx_buffer[idx_buffer >= 0]

    visible_indices = np.unique(visible_indices)
    visible_indices.sort()

    pixels = np.round(pts_uvz[visible_indices, :2]).astype(int)

    return pixels, visible_indices

from scipy.interpolate import splprep, splev
import numpy as np

def fit_bspline_curve(points, smooth=0):
    x, y = points[:,0], points[:,1]
    tck, _ = splprep([x, y], s=smooth, k=3)
    return tck

def epipolar_line(F, pt):
    """
    pt: (u,v)
    return: ax + by + c = 0
    """
    x = np.array([pt[0], pt[1], 1.0])
    l = F @ x
    return l / np.linalg.norm(l[:2])

def intersect_line_bspline(line, tck, num=2000):
    """
    line: [a,b,c]
    return: list of intersection points
    """
    a, b, c = line

    ts = np.linspace(0, 1, num)
    xs, ys = splev(ts, tck)
    vals = a*xs + b*ys + c

    pts = []

    for i in range(len(vals)-1):
        if vals[i] == 0:
            pts.append((xs[i], ys[i]))
        elif vals[i]*vals[i+1] < 0:
            t = abs(vals[i]) / (abs(vals[i]) + abs(vals[i+1]))
            x = xs[i]*(1-t) + xs[i+1]*t
            y = ys[i]*(1-t) + ys[i+1]*t
            pts.append((x,y))

    return pts

def match_curve_monotonic(pts_L, F, tck_R):

    matched_R = []
    prev_pt = None

    for pt in pts_L:

        line = epipolar_line(F, pt)
        candidates = intersect_line_bspline(line, tck_R)

        if len(candidates) == 0:
            matched_R.append(None)
            continue

        if prev_pt is None:
            chosen = min(candidates, key=lambda p: p[0])
        else:
            chosen = min(
                candidates,
                key=lambda p: np.linalg.norm(np.array(p)-np.array(prev_pt))
            )

        matched_R.append(chosen)
        prev_pt = chosen

    return matched_R

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

def find_duplicate_segments_idx(idx_L):
    segments = []
    n = len(idx_L)
    start = 0
    while start < n:
        end = start + 1

        while end < n and idx_L[end] == idx_L[start]:
            end += 1

        if end - start > 1:
            segments.append((start, end))

        start = end

    return segments

def ecdp(seq_L, seq_R, F, P1, P2, refine = False):
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

    matches = []
    i, j = N1-1, N2-1
    while i != -1 and j != -1:
        matches.append((i,j))
        pi, pj = parent[i,j]
        i, j = pi, pj
    matches.reverse()

    pts_3d = []
    pts_L = []
    pts_R = []
    idx_L = []
    idx_R = []
    for match in matches:
        pt_L = np.array(seq_L[match[0]][::-1], dtype=np.float32).reshape(2, 1)
        pt_R = np.array(seq_R[match[1]][::-1], dtype=np.float32).reshape(2, 1)
        if not refine:
            points4D = cv2.triangulatePoints(P1, P2, pt_L, pt_R)
            pt_3d = (points4D[:3] / points4D[3]).flatten()
            pts_3d.append(pt_3d)
        else:
            pts_L.append(pt_L)
            pts_R.append(pt_R)
            idx_L.append(match[0])
            idx_R.append(match[1])
    if refine:
        idx_L_repeat = find_duplicate_segments_idx(idx_L)
        for seg in idx_L_repeat:
            idx_seg_L = idx_L[seg[0]:seg[1]]
            idx_seg_R = idx_R[seg[0]:seg[1]]
            seg_D = D[[idx_seg_L[0]]][:, idx_seg_R][0]
            j_s = np.argmin(seg_D, axis=0)
            for j in range(0, j_s):
                if idx_seg_L[j] == 0: break
                pt_L_i = pts_L[seg[0]+j]
                pt_L_im1 = pts_L[seg[0]+j-1]
                ij = (idx_seg_L[j], idx_seg_R[j])
                D_ij = D[ij]
                D_im1j = D[idx_seg_L[j]-1, idx_seg_R[j]]
                pt_L_eff = D_ij/(D_ij+D_im1j)*pt_L_im1 + D_im1j/(D_ij+D_im1j)*pt_L_i
                pts_L[seg[0]+j] = pt_L_eff
            for j in range(j_s+1, seg[1]-seg[0]):
                if idx_seg_L[j] == N1-1: break
                pt_L_i = pts_L[seg[0]+j]
                pt_L_ip1 = pts_L[seg[0]+j+1]
                ij = (idx_seg_L[j], idx_seg_R[j])
                D_ij = D[ij]
                D_ip1j = D[idx_seg_L[j]+1, idx_seg_R[j]]
                pt_L_eff = D_ij/(D_ij+D_ip1j)*pt_L_ip1 + D_ip1j/(D_ij+D_ip1j)*pt_L_i
                pts_L[seg[0]+j] = pt_L_eff
        for i in range(len(pts_L)):
            pt_L = pts_L[i]
            pt_R = pts_R[i]
            points4D = cv2.triangulatePoints(P1, P2, pt_L, pt_R)
            pt_3d = (points4D[:3] / points4D[3]).flatten()
            pts_3d.append(pt_3d)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_3d)
    return pcd

def to_homogeneous(pts):
    return np.hstack([pts, np.ones((pts.shape[0], 1))])

def build_candidates(left_pts, right_pts, F, threshold=1.0):
    left_h = to_homogeneous(left_pts)
    right_h = to_homogeneous(right_pts)

    candidates = []

    for i in range(len(left_pts)):
        l = F @ left_h[i]

        a, b, c = l
        norm = np.sqrt(a*a + b*b)

        for j in range(len(right_pts)):
            dist = abs(right_h[j] @ l) / norm
            if dist < threshold:
                candidates.append((i, j))

    return candidates

def longest_monotonic_matching(candidates):
    candidates = sorted(candidates)

    n = len(candidates)
    dp = [1] * n
    parent = [-1] * n

    for i in range(n):
        for j in range(i):
            if (candidates[j][0] < candidates[i][0] and
                candidates[j][1] < candidates[i][1]):
                if dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
                    parent[i] = j

    idx = np.argmax(dp)
    seq = []

    while idx != -1:
        seq.append(candidates[idx])
        idx = parent[idx]

    return seq[::-1]

def triangulate_points(matches, left_pts, right_pts, P1, P2):
    points_3d = []

    for i, j in matches:
        x1 = np.append(left_pts[i], 1)
        x2 = np.append(right_pts[j], 1)

        A = np.array([
            x1[0]*P1[2]-P1[0],
            x1[1]*P1[2]-P1[1],
            x2[0]*P2[2]-P2[0],
            x2[1]*P2[2]-P2[1]
        ])

        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X /= X[3]

        points_3d.append(X[:3])

    return np.array(points_3d)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--category',
        type=str,
        default='simulation',
        help='Dataset category: simulation or phantom (default: simulation)'
    )

    parser.add_argument(
        '--method',
        type=str,
        default='ACCURATE',
        help='Reconstruction method: ACCURATE/TMI03/TMI15 (default: ACCURATE)'
    )

    parser.add_argument(
        '--refine',
        action='store_true',
        help='Enable refinement step (default: False)'
    )

    parser.add_argument(
        '--save_path',
        type=str,
        default='experiment',
        help='Save path for reconstruction results (default: experiment)'
    )
    
    args = parser.parse_args()
    category = args.category
    method = args.method
    refine = args.refine
    save_path = args.save_path

    if category == 'simulation':
        H,W = 2048, 512
    elif category == 'phantom':
        H,W = 500, 500

    data_path = f'ACCURATE_dataset/{category}'
    test_names = []
    with open(f'ACCURATE_dataset/splits/{category}_test.txt', 'r', encoding='utf-8') as f:
        fileline = f.readline()
        while fileline:
            test_names.append(fileline.strip())
            fileline = f.readline()

    for test_name in test_names:
        # print(test_name)
        test_path = os.path.join(data_path, test_name)

        pts_3d = o3d.io.read_point_cloud(f'ACCURATE_dataset/{category}/{test_name}/annotations/guidewire_3D.ply')
        pts_3d = np.asarray(pts_3d.points)

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

        pts_uvz_L = project_with_depth(pts_3d, K_L, R_L, t_L)
        pts_2d_L, idx_L = ordered_visible_pixels(pts_uvz_L, H, W)
        pts_uvz_R = project_with_depth(pts_3d, K_R, R_R, t_R)
        pts_2d_R, idx_R = ordered_visible_pixels(pts_uvz_R, H, W)

        if category == 'phantom':
            img_L = np.zeros((H,W))
            for pt in pts_2d_L:
                img_L[pt[1], pt[0]] = 255
            from skimage.morphology import skeletonize
            img_L = skeletonize(img_L>0).astype(np.uint8)
            y_L, x_L = np.where(img_L>0)

            img_R = np.zeros((H,W))
            for pt in pts_2d_R:
                img_R[pt[1], pt[0]] = 255
            from skimage.morphology import skeletonize
            img_R = skeletonize(img_R>0).astype(np.uint8)
            y_R, x_R = np.where(img_R>0)

            pts_2d_L = np.array(traverse_curve(img_L, start_point=(y_L[-1], x_L[-1]), r_min=10, r_max=50))[:,::-1]
            pts_2d_R = np.array(traverse_curve(img_R, start_point=(y_R[-1], x_R[-1]), r_min=10, r_max=50))[:,::-1]


        if method == 'TMI03':
            tck_R = fit_bspline_curve(pts_2d_R)
            matches_R = match_curve_monotonic(pts_2d_L, F, tck_R)

            pts_3d_rec = []
            for i in range(len(pts_2d_L)):
                if matches_R[i] is None:
                    continue
                pt_l = np.array(pts_2d_L[i], dtype=np.float32).reshape(2, 1)
                pt_r = np.array(matches_R[i], dtype=np.float32).reshape(2, 1)
            
                points4D = cv2.triangulatePoints(P1, P2, pt_l, pt_r)
                pt_prev_3d = (points4D[:3] / points4D[3]).flatten()

                pts_3d_rec.append(pt_prev_3d)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts_3d_rec)
        
        elif method == 'TMI15':
            candidates = build_candidates(pts_2d_L, pts_2d_R, F, threshold=1.5)

            matches = longest_monotonic_matching(candidates)

            P1 = K_L @ np.hstack([R_L, t_L.reshape(-1,1)])
            P2 = K_R @ np.hstack([R_R, t_R.reshape(-1,1)])

            points_3d = triangulate_points(matches, pts_2d_L, pts_2d_R, P1, P2)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_3d)
        
        elif method == 'ACCURATE':
            pts_2d_L = pts_2d_L[:,::-1]
            pts_2d_R = pts_2d_R[:,::-1]
            pcd = ecdp(pts_2d_L, pts_2d_R, F, P1, P2, refine)
        
        o3d.io.write_point_cloud(f"{save_path}/point/{category}/{method}/{test_name}.ply", pcd)