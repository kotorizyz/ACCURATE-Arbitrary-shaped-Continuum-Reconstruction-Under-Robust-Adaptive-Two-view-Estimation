import numpy as np
import cv2
from skimage.morphology import skeletonize
from skimage.draw import line
from scipy import linalg
import torch
import open3d as o3d

from skimage.morphology import skeletonize_3d

def project_points(points_xyz, P):
    N = points_xyz.shape[0]
    pts_h = np.hstack([points_xyz, np.ones((N,1))])
    proj = (P @ pts_h.T).T

    u = proj[:, 0] / proj[:, 2]
    v = proj[:, 1] / proj[:, 2]
    return np.stack([u, v], axis=1)

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
    
    # P1 = K1 @ np.hstack((np.eye(3), np.zeros((3,1))))
    # P2 = K2 @ np.hstack((R, T.reshape(3,1)))

    P1 = K1 @ np.concatenate((R1, t1[:,None]), axis=1)
    P2 = K2 @ np.concatenate((R2, t2[:,None]), axis=1)
    return R, T, E, F, P1, P2

#==================================
# CHECK FUNCTION
def draw_epiline(img, line):
    a, b, c = line
    h, w = img.shape[:2]
    x0, y0 = 0, int(-c / b) if b != 0 else 0
    x1, y1 = w, int(-(c + a * w) / b) if b != 0 else h
    return cv2.line(img, (x0,y0), (x1,y1), 1, 1)

def points_to_strong_contrast_image(
    points,
    H,
    W,
    radius=2,
):
    """
    Render ordered points with a strong-contrast, small-figure-friendly
    cold-to-warm gradient.
    """

    points = np.asarray(points, dtype=np.int32)
    assert points.ndim == 2 and points.shape[1] == 2

    img = np.zeros((H, W, 3), dtype=np.uint8)
    N = len(points)
    if N == 0:
        return img

    # Strong-contrast BGR colors (small-figure optimized)
    c_start = np.array([140,  60,  20], dtype=np.float32)   # dark cool
    c_mid   = np.array([245, 245, 245], dtype=np.float32)  # very bright
    c_end   = np.array([ 20,  90, 220], dtype=np.float32)  # dark warm

    for i, (x, y) in enumerate(points):
        if not (0 <= x < W and 0 <= y < H):
            continue

        t = i / max(N - 1, 1)

        if t <= 0.5:
            alpha = t / 0.5
            color = (1 - alpha) * c_start + alpha * c_mid
        else:
            alpha = (t - 0.5) / 0.5
            color = (1 - alpha) * c_mid + alpha * c_end

        color = tuple(int(v) for v in color)
        img[y,x] = color

    return img

#==========================================

def fix_single_pixel_gaps(mask, max_gap=1):
    skel = skeletonize(mask>0).astype(np.uint8)

    H, W = skel.shape
    points = np.argwhere(skel>0)
    deg_map = np.zeros_like(skel, dtype=int)
    for x, y in points:
        deg = 0
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                if dx==0 and dy==0:
                    continue
                nx, ny = x+dx, y+dy
                if 0<=nx<H and 0<=ny<W and skel[nx,ny]:
                    deg += 1
        deg_map[x,y] = deg

    end_points = [tuple(p) for p in points if deg_map[p[0],p[1]]==1]

    for i, p1 in enumerate(end_points):
        for j, p2 in enumerate(end_points):
            if i >= j:
                continue
            dist = max(abs(p1[0]-p2[0]), abs(p1[1]-p2[1]))
            if 1 <= dist <= max_gap:
                rr, cc = line(p1[0], p1[1], p2[0], p2[1])
                skel[rr, cc] = 1

    return skel.astype(np.uint8)

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
    # if len(points)>5:
    #     pts = pts[:5]
    #     t = np.linspace(0,1,len(pts))
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

def fix_skel(mask):
    mask_fixed = fix_single_pixel_gaps(mask=mask, max_gap=10)
    skel = skeletonize(mask_fixed>0).astype(np.uint8)
    return skel

def traverse_curve(mask, window_size=10, r_min=1, r_max=2, gap_threshold=3):
    points = np.argwhere(mask>0)
    deg_map = np.zeros_like(mask, dtype=int)
    for x, y in points:
        deg_map[x, y] = len(get_candidates(mask, (x,y), set(), r_min=1, r_max=1))

    end_points = [tuple(p) for p in points if deg_map[p[0], p[1]]==2]
    # print(end_points)
    start_point = end_points[-1] if len(end_points)>0 else tuple(points[0])

    visited = set()
    sequence = [start_point]
    visited.add(start_point)
    prev_point = None
    curr_point = start_point

    eps = 1e-4
    while True:
        candidates = get_candidates(mask, curr_point, visited, r_min=r_min, r_max=r_max)
        if len(candidates)==0:
            # print(curr_point)
            # quit()
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
                dist_penalty = dist
                score = k + angle_penalty + 0.5*dist_penalty
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
                    
            #     if(deg_map[curr_point] >= 4):
            #         print(k, angle_penalty, dist_penalty, p, score)
            # if(deg_map[curr_point] >= 4):
            #     print(curr_point, "=============")
        # print(curr_point, next_point)
        sequence.append(next_point)
        visited.add(next_point)
        prev_point = curr_point
        curr_point = next_point

    # all_points = [tuple(p) for p in np.argwhere(skel>0)]
    # unvisited = [p for p in all_points if p not in visited]
    # for p in unvisited:
    #     sequence.append(p)
    #     visited.add(p)
    return sequence


def dp_left_right_matching(epi, tau=3):
    N1, N2 = epi.shape

    candidates = [[] for _ in range(N1)]
    for i in range(N1):
        row = epi[i]
        for j in range(N2-1):
            v0, v1 = row[j], row[j+1]
            if v0 * v1 <= 0:
                denom = v1 - v0
                t = -v0 / denom if denom != 0 else 0.5
                j_frac = j + np.clip(t, 0.0, 1.0)
                score = -abs(v0 + t*(v1-v0))
                candidates[i].append((j_frac, score))
        small_idxs = np.where(np.abs(row) <= tau)[0]
        for j in small_idxs:
            candidates[i].append((float(j), -abs(row[j])))

        if len(candidates[i]) == 0:
            j_min = np.argmin(np.abs(row))
            candidates[i].append((float(j_min), -abs(row[j_min])))

        candidates[i].sort(key=lambda x: x[0])

    dp = [dict() for _ in range(N1)]
    for k, (j, score) in enumerate(candidates[0]):
        dp[0][k] = (score, -1)

    for i in range(1, N1):
        print(i, candidates[i])
        for k, (j, score) in enumerate(candidates[i]):
            best = None
            for prev_k, (prev_score, _) in dp[i-1].items():
                prev_j = candidates[i-1][prev_k][0]
                if j >= prev_j:
                    total_score = prev_score + score
                    if (best is None) or (total_score > best[0]):
                        best = (total_score, prev_k)
            if best is None:
                if len(dp[i-1]) > 0:
                    prev_k, (prev_score, _) = min(
                        dp[i-1].items(),
                        key=lambda item: abs(candidates[i][k][0] - candidates[i-1][item[0]][0])
                    )
                    total_score = prev_score + score
                    best = (total_score, prev_k)
                else:
                    best = (score, -1)
            dp[i][k] = best


    if len(dp[N1-1]) == 0:
        return []
    last_k = max(dp[N1-1], key=lambda k: dp[N1-1][k][0])
    matches = []
    for i in reversed(range(N1)):
        j = candidates[i][last_k][0]
        matches.append((i, int(round(j))))
        _, last_k = dp[i][last_k]
        if last_k == -1:
            break
    matches.reverse()
    return matches

def min_cost_path_and_route(D):
    N1, N2 = D.shape
    
    dp = np.zeros_like(D, dtype=float)
    parent = np.full((N1, N2, 2), -1, dtype=int)

    dp[0,0] = D[0,0]
    for j in range(1, N2):
        dp[0,j] = dp[0,j-1] + D[0,j]
        parent[0,j] = [0, j-1]
    for i in range(1, N1):
        dp[i,0] = dp[i-1,0] + D[i,0]
        parent[i,0] = [i-1, 0]
    for i in range(1, N1):
        for j in range(1, N2):
            if dp[i-1,j] < dp[i,j-1]:
                dp[i,j] = dp[i-1,j] + D[i,j]
                parent[i,j] = [i-1, j]
            else:
                dp[i,j] = dp[i,j-1] + D[i,j]
                parent[i,j] = [i, j-1]

    path = []
    i, j = N1-1, N2-1
    while i != -1 and j != -1:
        path.append((i,j))
        pi, pj = parent[i,j]
        i, j = pi, pj

    path.reverse()
    return dp[-1,-1], path

if __name__ == '__main__':

    H, W = 500,500

    mask1 = np.load('mask/data0_gt_106.npz')
    fusion1 = cv2.imread('tmp/gt_106.png', cv2.IMREAD_GRAYSCALE)
    # fusion1[mask1['probabilities'][1,0] > 0.5] = 255
    # cv2.imwrite('fusionL.png', fusion1)
    mask1 = (mask1['probabilities'][1,0] > 0.5) * 1.0
    mask2 = np.load('tmp/data0_gt_168.npz')
    # fusion2 = cv2.imread('tmp/gt_168.png', cv2.IMREAD_GRAYSCALE)
    # fusion2[mask2['probabilities'][1,0] > 0.5] = 255
    # cv2.imwrite('fusionR.png', fusion2)
    mask2 = (mask2['probabilities'][1,0] > 0.5) * 1.0
    # quit()

    pts_order_L = []
    mask1 = fix_skel(mask1)
    seq = traverse_curve(mask1, window_size=10, r_min=10, r_max=15, gap_threshold=3)
    pts_order_L.append(seq)

    # mask = np.zeros((H,W))
    # for i in range(len(seq)):
    #     mask[seq[i][0], seq[i][1]] = 255
    #     if i % 10 == 0:
    #         cv2.imwrite(f"order/{i}.png", mask)
    # quit()

    pts_order_R = []
    mask2 = fix_skel(mask2)
    seq = traverse_curve(mask2, window_size=10, r_min=10, r_max=15, gap_threshold=3)
    pts_order_R.append(seq)

    # seq_show = np.zeros((len(seq),2))
    # for i in range(len(seq)):
    #     seq_show[i] = np.array([seq[i][1], seq[i][0]], dtype=np.int32)
    # print(seq_show)
    # img_show = points_to_strong_contrast_image(seq_show, H, W)
    # cv2.imwrite('curve_order_R.png', img_show)
    # quit()

    # List: pts_order_L, pts_order_R, param
    for num_guidewire in range(len(pts_order_L)):
        pts_L = [[pts_order_L[num_guidewire][i][0], pts_order_L[num_guidewire][i][1]] for i in range(len(pts_order_L[num_guidewire]))]
        pts_R = [[pts_order_R[num_guidewire][i][0], pts_order_R[num_guidewire][i][1]] for i in range(len(pts_order_R[num_guidewire]))]
        pts_L = np.array(pts_L)
        pts_R = np.array(pts_R)
        prev_idx_L = 0
        prev_idx_R = 0
        pts_3d = []

        K1 = np.load('tmp/K_106.npy')
        R1 = np.load('tmp/Rt_106.npy')[:,:3]
        t1 = np.load('tmp/Rt_106.npy')[:,3]
        K2 = np.load('tmp/K_168.npy')
        R2 = np.load('tmp/Rt_168.npy')[:,:3]
        t2 = np.load('tmp/Rt_168.npy')[:,3]
        R, T, E, F, P1, P2 = compute_stereo_params(K1,K2,R1,R2,t1,t2)

        line_right = cv2.computeCorrespondEpilines(pts_L[:,::-1], 1, F).reshape(-1, 3)
        N1 = line_right.shape[0]
        N2 = pts_R.shape[0]
        epi = np.zeros((N1, N2), dtype=np.float32)
        for i in range(N1):
            a, b, c = line_right[i]
            xs = pts_R[:,1]
            ys = pts_R[:,0]
            num = a*xs + b*ys + c
            denom = np.sqrt(a*a + b*b)
            epi[i] = num / denom
        # matches = dp_left_right_matching(epi)
        cost, matches = min_cost_path_and_route(np.abs(epi))

        idx_L_prev = 0
        idx_R_prev = 0
        max_dlt = 0
        num_pair = len(matches)
        
        for i in range(num_pair):
            # if(matches[i][0] >= pts_L.shape[0]-10 or matches[i][1] >= pts_R.shape[0]-10):
            #     break
            idx_L = matches[i][0]
            idx_R = matches[i][1]
            # dlt_L = idx_L - idx_L_prev
            # dlt_R = idx_R - idx_R_prev

            # if dlt_R > 1:
            #     for j in range(1, dlt_R):
            #         idx_R_interp = idx_R_prev + j
            #         pt_l = np.array(pts_L[idx_L][::-1], dtype=np.float32).reshape(2, 1)
            #         pt_r = np.array(pts_R[idx_R_interp][::-1], dtype=np.float32).reshape(2, 1)
            #         points4D = cv2.triangulatePoints(P1, P2, pt_l, pt_r)
            #         pt_prev_3d = (points4D[:3] / points4D[3]).flatten()
            #         pts_3d.append(pt_prev_3d)

            pt_l = np.array(pts_L[idx_L][::-1], dtype=np.float32).reshape(2, 1)
            pt_r = np.array(pts_R[idx_R][::-1], dtype=np.float32).reshape(2, 1)
            
            points4D = cv2.triangulatePoints(P1, P2, pt_l, pt_r)
            pt_prev_3d = (points4D[:3] / points4D[3]).flatten()

            pts_3d.append(pt_prev_3d)
            idx_L_prev = idx_L
            idx_R_prev = idx_R
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_3d)
        o3d.io.write_point_cloud(f"output/real/rec_0.ply", pcd)

        np.save('pts.npy', pts_3d)
        quit()

        pts_3d = np.array(pts_3d)
        pts_2d = project_points(pts_3d, P2)
        img = np.zeros((500,500))
        for pt_2d in pts_2d:
            img[np.int32(pt_2d[1]), np.int32(pt_2d[0])] = 255
        cv2.imwrite('rec.png', img)

        gt = o3d.io.read_point_cloud(f'output/real/gt_0.ply')
        gt = np.asarray(gt.points)
        pts_2d = project_points(gt, P2)
        img = np.zeros((500,500))
        for pt_2d in pts_2d:
            img[np.int32(pt_2d[1]), np.int32(pt_2d[0])] = 255
        cv2.imwrite('gt.png', img)