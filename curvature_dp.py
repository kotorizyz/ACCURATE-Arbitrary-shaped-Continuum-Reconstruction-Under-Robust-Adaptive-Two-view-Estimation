import numpy as np
import cv2
from skimage.morphology import skeletonize
from skimage.draw import line
from scipy import linalg
import torch
import open3d as o3d

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

# CHECK FUNCTION
def draw_epiline(img, line):
    a, b, c = line
    h, w = img.shape[:2]
    x0, y0 = 0, int(-c / b) if b != 0 else 0
    x1, y1 = w, int(-(c + a * w) / b) if b != 0 else h
    return cv2.line(img, (x0,y0), (x1,y1), 1, 1)


# def fix_single_pixel_gaps(mask, max_gap=1):
#     skel = skeletonize(mask>0).astype(np.uint8)

#     H, W = skel.shape
#     points = np.argwhere(skel>0)
#     deg_map = np.zeros_like(skel, dtype=int)
#     for x, y in points:
#         deg = 0
#         for dx in [-1,0,1]:
#             for dy in [-1,0,1]:
#                 if dx==0 and dy==0:
#                     continue
#                 nx, ny = x+dx, y+dy
#                 if 0<=nx<H and 0<=ny<W and skel[nx,ny]:
#                     deg += 1
#         deg_map[x,y] = deg

#     end_points = [tuple(p) for p in points if deg_map[p[0],p[1]]==1]

#     for i, p1 in enumerate(end_points):
#         for j, p2 in enumerate(end_points):
#             if i >= j:
#                 continue
#             dist = max(abs(p1[0]-p2[0]), abs(p1[1]-p2[1]))
#             if 1 <= dist <= max_gap:
#                 rr, cc = line(p1[0], p1[1], p2[0], p2[1])
#                 skel[rr, cc] = 1

#     return skel.astype(np.uint8)

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

def traverse_curve(mask, window_size=10, r_min=1, r_max=2, gap_threshold=3):
    # mask_fixed = fix_single_pixel_gaps(mask)

    skel = skeletonize(mask>0).astype(np.uint8)
    # cv2.imwrite('skel.png', skel*255)

    points = np.argwhere(skel>0)
    deg_map = np.zeros_like(skel, dtype=int)
    for x, y in points:
        deg_map[x, y] = len(get_candidates(skel, (x,y), set(), r_min=1, r_max=1))

    end_points = [tuple(p) for p in points if deg_map[p[0], p[1]]==2]
    # print(end_points)
    start_point = end_points[0] if len(end_points)>0 else tuple(points[0])

    visited = set()
    sequence = [start_point]
    visited.add(start_point)
    prev_point = None
    curr_point = start_point

    eps = 1e-4
    while True:
        candidates = get_candidates(skel, curr_point, visited, r_min=r_min, r_max=r_max)
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


def dp_left_right_matching(epi, tau=2):
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



if __name__ == '__main__':

    H, W = 2048, 512
    num_rec = 1
    epsilon = 3
    pts_3d_gt = []
    pts_L = []
    pts_R = []
    param = []
    for i in range(num_rec):
        data_i = torch.load(f'./dataset/processed_data/data_{i+2}.pt')
        img = np.zeros((H, W))

        uv1 = np.rint(data_i['uv1']).numpy().astype(np.int32)
        uv2 = np.rint(data_i['uv2']).numpy().astype(np.int32)
        pts_L.append(uv1)
        pts_R.append(uv2)

        K1 = data_i['K1'].numpy()
        K2 = data_i['K2'].numpy()
        R1 = data_i['RT1'][:3,:3].numpy()
        R2 = data_i['RT2'][:3,:3].numpy()
        t1 = data_i['RT1'][:3,3].numpy()
        t2 = data_i['RT2'][:3,3].numpy()
        param.append({'K1': K1, 'K2': K2, 'R1': R1, 'R2': R2, 't1': t1, 't2': t2})

        pts_3d_gt.append(data_i['points'].numpy())

    pts_order_L = []
    for i in range(len(pts_L)):
        img = np.zeros((H,W))
        # img[pts_L[i][:,1], pts_L[i][:,0]] = 1
        for j in range(len(pts_L[i]) - 1):
            x1, y1 = pts_L[i][j]
            x2, y2 = pts_L[i][j+1]
            cv2.line(img, (x1, y1), (x2, y2), color=1, thickness=1)

        seq = traverse_curve(img, window_size=10, r_min=10, r_max=15, gap_threshold=3)
        # mask = np.zeros((H,W))
        # for i in range(len(seq)):
        #     mask[seq[i][0], seq[i][1]] = 255
        #     if i % 10 == 0:
        #         cv2.imwrite(f"order/{i}.png", mask)
        # quit()
        # print(seq)
        pts_order_L.append(seq)
        # print(len(seq))
        # print(len(pts_order_L), len(pts_order_L[0]))

    pts_order_R = []
    for i in range(len(pts_R)):
        img = np.zeros((H,W))
        # img[pts_R[i][:,1], pts_R[i][:,0]] = 1
        for j in range(len(pts_R[i]) - 1):
            x1, y1 = pts_R[i][j]
            x2, y2 = pts_R[i][j+1]
            cv2.line(img, (x1, y1), (x2, y2), color=1, thickness=1)
        # img[1055, 175] = 2
        # cv2.imwrite('cam2.png', img*120)
        # quit()
        seq = traverse_curve(img, window_size=10, r_min=10, r_max=15, gap_threshold=3)
        # mask = np.zeros((H,W))
        # for i in range(len(seq)):
        #     mask[seq[i][0], seq[i][1]] = 255
        #     if i % 10 == 0:
        #         cv2.imwrite(f"order/{i}.png", mask)
        # quit()
        # print(seq)
        pts_order_R.append(seq)
        # print(len(seq))
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

        K1 = param[num_guidewire]['K1']
        K2 = param[num_guidewire]['K2']
        R1 = param[num_guidewire]['R1']
        R2 = param[num_guidewire]['R2']
        t1 = param[num_guidewire]['t1']
        t2 = param[num_guidewire]['t2']
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
        
        matches = dp_left_right_matching(epi)

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
        o3d.io.write_point_cloud(f"output/rec_{num_guidewire}.ply", pcd)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_3d_gt[num_guidewire])
        o3d.io.write_point_cloud(f"output/gt_{num_guidewire}.ply", pcd)