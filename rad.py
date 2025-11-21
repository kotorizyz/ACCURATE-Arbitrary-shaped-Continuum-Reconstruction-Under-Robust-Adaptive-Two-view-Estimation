import numpy as np
import cv2
from skimage.morphology import skeletonize
from skimage.draw import line
from scipy import linalg
import torch

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

    P1 = K1 @ np.concatenate((K1, R1), axis=1)
    P2 = K2 @ np.concatenate((K2, R2), axis=1)
    print(np.concatenate((K1, R1), axis=1))
    quit()
    
    return R, T, E, F, P1, P2

def remove_duplicates_keep_last(arr):
    seen = set()
    output = []

    for x, y in arr[::-1]:
        key = (int(x), int(y))
        if key not in seen:
            seen.add(key)
            output.append([x, y])

    output.reverse()
    return np.array(output, dtype=arr.dtype)

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
    for dx in range(-r_max, r_max+1):
        for dy in range(-r_max, r_max+1):
            nx, ny = curr[0]+dx, curr[1]+dy
            if 0 <= nx < H and 0 <= ny < W and mask[nx, ny]>0 and (nx, ny) not in visited:
                dist = max(abs(dx), abs(dy))
                candidates.append(((nx, ny), dist))
    return candidates

def fit_curve(points):
    pts = np.array(points)
    if len(points)<3:
        return np.zeros(3), np.zeros(3)
    t = np.linspace(0,1,len(points))
    A = np.vstack([t**2, t, np.ones_like(t)]).T
    coef_x, _, _, _ = linalg.lstsq(A, pts[:,0])
    coef_y, _, _, _ = linalg.lstsq(A, pts[:,1])
    return coef_x, coef_y

def curvature_at_t(coef_x, coef_y, t):
    a_x, b_x, _ = coef_x
    a_y, b_y, _ = coef_y
    x_prime = 2*a_x*t + b_x
    x_double = 2*a_x
    y_prime = 2*a_y*t + b_y
    y_double = 2*a_y
    k = abs(x_prime*y_double - y_prime*x_double) / ((x_prime**2 + y_prime**2)**1.5 + 1e-8)
    return k

def traverse_curve(mask, window_size=5, r_min=1, r_max=2, gap_threshold=3):
    mask_fixed = fix_single_pixel_gaps(mask)

    skel = skeletonize(mask_fixed>0).astype(np.uint8)

    points = np.argwhere(skel>0)
    deg_map = np.zeros_like(skel, dtype=int)
    for x, y in points:
        deg_map[x, y] = len(get_candidates(skel, (x,y), set(), r_min=1, r_max=1))
    end_points = [tuple(p) for p in points if deg_map[p[0], p[1]]==1]
    start_point = end_points[0] if len(end_points)>0 else tuple(points[0])

    visited = set()
    sequence = [start_point]
    visited.add(start_point)
    prev_point = None
    curr_point = start_point

    while True:
        candidates = get_candidates(skel, curr_point, visited, r_min=r_min, r_max=r_max)
        if len(candidates)==0:
            break
        elif len(candidates)==1:
            next_point = candidates[0][0]
        else:
            window = sequence[-window_size:]
            min_score = float('inf')
            next_point = candidates[0][0]
            for p, dist in candidates:
                pts = window + [p]
                coef_x, coef_y = fit_curve(pts)
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
                    min_score = score
                    next_point = p
        sequence.append(next_point)
        visited.add(next_point)
        prev_point = curr_point
        curr_point = next_point

    all_points = [tuple(p) for p in np.argwhere(skel>0)]
    unvisited = [p for p in all_points if p not in visited]
    for p in unvisited:
        sequence.append(p)
        visited.add(p)
    return sequence

H, W = 2048, 512
num_train = 1
pts_L = []
pts_R = []
param = []
for i in range(num_train):
    data_i = torch.load(f'./dataset/processed_data/data_{i+1}.pt')
    img = np.zeros((H, W))

    uv1 = np.rint(data_i['uv1']).numpy().astype(np.int32)
    uv1 = remove_duplicates_keep_last(uv1)
    uv2 = np.rint(data_i['uv2']).numpy().astype(np.int32)
    uv2 = remove_duplicates_keep_last(uv2)
    pts_L.append(uv1)
    pts_R.append(uv2)

    K1 = data_i['K1'].numpy()
    K2 = data_i['K2'].numpy()
    R1 = data_i['RT1'][:3,:3].numpy()
    R2 = data_i['RT2'][:3,:3].numpy()
    t1 = data_i['RT1'][:3,3].numpy()
    t2 = data_i['RT2'][:3,3].numpy()
    param.append({'K1': K1, 'K2': K2, 'R1': R1, 'R2': R2, 't1': t1, 't2': t2})

pts_order_L = []
for i in range(len(pts_L)):
    img = np.zeros((H,W))
    img[pts_L[i][:,1], pts_L[i][:,0]] = 1
    seq = traverse_curve(img, window_size=5, r_min=5, r_max=10, gap_threshold=3)
    # mask = np.zeros((H,W))
    # for i in range(len(seq)):
    #     mask[seq[i][0], seq[i][1]] = 255
    #     if i % 50 == 0:
    #         cv2.imwrite(f"{i / 25}.png", mask)
    # quit()
    pts_order_L.append(seq)
    # print(len(pts_order_L), len(pts_order_L[0]))

pts_order_R = []
for i in range(len(pts_R)):
    img = np.zeros((H,W))
    img[pts_R[i][:,1], pts_R[i][:,0]] = 1
    seq = traverse_curve(img, window_size=5, r_min=5, r_max=10, gap_threshold=3)
    # mask = np.zeros((H,W))
    # for i in range(len(seq)):
    #     mask[seq[i][0], seq[i][1]] = 255
    #     if i % 50 == 0:
    #         cv2.imwrite(f"{i / 25}.png", mask)
    # quit()
    pts_order_R.append(seq)
    # print(len(pts_order_R), len(pts_order_R[0]))

# List: pts_order_L, pts_order_R, param
for num_guidewire in range(len(pts_order_L)):
    prev_idx_L = -1
    prev_idx_R = -1
    for idx_L in range(len(pts_order_L[num_guidewire])):
        uv_L = [pts_order_L[num_guidewire][0], pts_order_L[num_guidewire][1]]
        K1 = param[num_guidewire]['K1']
        K2 = param[num_guidewire]['K2']
        R1 = param[num_guidewire]['R1']
        R2 = param[num_guidewire]['R2']
        t1 = param[num_guidewire]['t1']
        t2 = param[num_guidewire]['t2']
        compute_stereo_params(K1,K2,R1,R2,t1,t2)