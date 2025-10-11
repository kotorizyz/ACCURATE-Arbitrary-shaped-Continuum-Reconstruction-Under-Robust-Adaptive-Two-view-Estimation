import cv2
import numpy as np
import json
import open3d as o3d
from skimage.morphology import skeletonize
from skimage.draw import line
from scipy.signal import convolve2d

def compute_stereo_params(cams):
    cam1_intrinsic = cams['cam1']['intrinsic']
    cam1_extrinsic = cams['cam1']['extrinsic']
    cam2_intrinsic = cams['cam2']['intrinsic']
    cam2_extrinsic = cams['cam2']['extrinsic']
    
    dist_l = np.zeros(5)
    dist_r = np.zeros(5)

    # K1 = np.array(cam1_intrinsic, dtype=float)
    # K2 = np.array(cam2_intrinsic, dtype=float)

    K1 = np.array([[1988, 0, 256],[0, 1988, 1024],[0,0,1]]).astype(np.float32)
    K2 = np.array([[1988, 0, 256],[0, 1988, 1024],[0,0,1]]).astype(np.float32)
    
    T1 = np.array(cam1_extrinsic, dtype=float)
    T2 = np.array(cam2_extrinsic, dtype=float)
    
    R1 = T1[:3,:3]
    t1 = T1[:3,3]
    R2 = T2[:3,:3]
    t2 = T2[:3,3]
    
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

    P1 = K1 @ np.array(cam1_extrinsic)[:3]
    P2 = K2 @ np.array(cam2_extrinsic)[:3]
    
    return K1, dist_l, K2, dist_r, R, T, E, F, P1, P2

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

def json_to_mask_ordered(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mask_points_ordered = []
    shape = data["shapes"][0]
    points = np.array(shape["points"], dtype=np.int32)
    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i+1]
        rr, cc = line(y0, x0, y1, x1)
        mask_points_ordered.extend(list(zip(rr[:-1], cc[:-1])))
    mask_points_ordered = np.array(mask_points_ordered, dtype=np.int32)
    return mask_points_ordered


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

def reconstruct_polyline_3d(mask1_points, mask2_points, P1, P2, F):
    pts_3d = []
    if(len(mask1_points)==0 or len(mask2_points)==0):
        return np.array(pts_3d)

    pt_prev_3d = (0,0,0)
    prev_idx = -1

    len_l = float(len(mask1_points))
    len_r = float(len(mask2_points))

    for idx in range(len(mask1_points)):
        idx_r = int((len_r / len_l) * idx)
        v_l, u_l = mask1_points[idx]
        pt_l = np.array([u_l, v_l], dtype=np.float32).reshape(2, 1)
        uv_point  = (mask2_points[idx_r][1], mask2_points[idx_r][0])
        pt_r = np.array(uv_point, dtype=np.float32).reshape(2, 1)
        points4D = cv2.triangulatePoints(P1, P2, pt_l, pt_r)
        pt_prev_3d = (points4D[:3] / points4D[3]).flatten()
        pts_3d.append(pt_prev_3d)
    return np.array(pts_3d)


    for (v_l, u_l) in mask1_points:
        line_right = cv2.computeCorrespondEpilines(np.array([[[u_l, v_l]]], dtype=np.float32), 1, F).reshape(-1, 3)[0]
        a, b, c = line_right
        ys, xs = mask2_points[:,0], mask2_points[:,1]
        dist = a*xs + b*ys + c / np.sqrt(a*a + b*b)
        pts_dist = list(zip(xs, ys, dist))
        

        # img = np.zeros((2048, 512))
        # for [v,u] in mask2_points:
        #     img[v,u] = 1
        # img = draw_epiline(img, line_right)
        # cv2.imwrite('1.png', img*255)
        # quit()

        idx_list = []
        for i in range(len(pts_dist)-1):
            if(pts_dist[i][2]*pts_dist[i+1][2]<0):
                idx_list.append(i)

        # if(len(idx_list)==1):
        #     idx_prev_right = idx_list[0]
        #     uv_prev_right = (pts_dist[idx_prev_right][0], pts_dist[idx_prev_right][1])
        # elif(len(idx_list)>1):
        #     for idx in idx_list:
        #         if(idx < idx_prev_right):
        #             continue
        #         uv_prev_right = (pts_dist[idx][0], pts_dist[idx][1])
        #         idx_prev_right = idx
        #         break
        #     uv_prev_right = (pts_dist[idx_prev_right][0], pts_dist[idx_prev_right][1])
        
        # print(uv_prev_right)

        if(len(idx_list)==0): continue
        print(idx_list, prev_idx)
        flag = False
        for idx in idx_list:
            if(idx < prev_idx): continue
            prev_idx = idx
            uv_point  = (mask2_points[idx][1], mask2_points[idx][0])
            pt_l = pt_l = np.array([u_l, v_l], dtype=np.float32).reshape(2, 1)
            pt_r = np.array(uv_point, dtype=np.float32).reshape(2, 1)
            # pt_l_ud = cv2.undistortPoints(np.expand_dims(pt_l, axis=1), mtx_l, dist_l)
            # pt_r_ud = cv2.undistortPoints(np.expand_dims(pt_r, axis=1), mtx_r, dist_r)
            points4D = cv2.triangulatePoints(P1, P2, pt_l, pt_r)
            pt_prev_3d = (points4D[:3] / points4D[3]).flatten()  # (x,y,z)
            flag = True
            break
        # print(pt_prev_3d)
        pts_3d.append(pt_prev_3d)

        if(prev_idx > 100): break

    return np.array(pts_3d)

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
# para = json.load(open('sim_data/calibration/camera_params.json'))
para = json.load(open('dataset/dataset_3/sample_0001/camera_params.json'))
mtx_l, dist_l, mtx_r, dist_r, R, T, E, F, P1, P2 = compute_stereo_params(para)
# mask1 = (cv2.imread("sim_data/mask/mask_1.png", cv2.IMREAD_GRAYSCALE) / 255).astype(np.uint8)
# mask2 = (cv2.imread("sim_data/mask/mask_2.png", cv2.IMREAD_GRAYSCALE) / 255).astype(np.uint8)
# mask_cam1 = (cv2.imread("sim_data/mask/mask_cam1.png", cv2.IMREAD_GRAYSCALE)).astype(np.uint8)
# mask_cam2 = (cv2.imread("sim_data/mask/mask_cam2.png", cv2.IMREAD_GRAYSCALE)).astype(np.uint8)

# mask1_points = json_to_mask_ordered("sim_data/mask/mask_cam1.json")
# mask2_points = json_to_mask_ordered("sim_data/mask/mask_cam2.json")
mask1_points = json_to_mask_ordered('dataset/dataset_3/sample_0001/mask_cam1.json')
mask2_points = json_to_mask_ordered('dataset/dataset_3/sample_0001/mask_cam2.json')


# img = np.zeros((2048, 512), dtype=np.uint8)
# for (v_l, u_l) in mask1_points:
#     img[v_l, u_l] += 1
# cv2.imwrite("mask1_points.png", img*100)
# quit()

# polyline_left = ordered_points_from_mask1(mask1)

# print(np.sum(mask1 > 1))
# print(len(polyline_left))

# img = np.zeros_like(mask2)
# for (u_l, v_l) in polyline_left:
#     img[u_l, v_l] = 1
# cv2.imwrite("polyline_left.png", img*255)
# quit()


# polyline_right = ordered_points_from_mask(mask2)

curve_3d= reconstruct_polyline_3d(mask1_points, mask2_points, P1, P2, F)
# curve_3d_r = reconstruct_polyline_3d(mask2_points, mask1_points, P2, P1, F.T)
# curve_3d = np.concatenate([curve_3d_l, curve_3d_r], axis=0)
print("3D points:", curve_3d.shape)

print(curve_3d[::-1])

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(curve_3d)
o3d.io.write_point_cloud("sim_data/reconstruction/guidewire.ply", pcd)

def project_points_with_P(points3D, P, image_size, point_size=1):
    N = points3D.shape[0]
    X_h = np.hstack((points3D, np.ones((N,1)))).T  # 4xN
    
    x_h = P @ X_h
    x = x_h[:2,:] / x_h[2,:]
    x = x.T
    
    h, w = image_size
    image = np.zeros((h, w), dtype=np.uint8)
    
    for u,v in x:
        u = int(round(u))
        v = int(round(v))
        if 0 <= u < w and 0 <= v < h:
            cv2.circle(image, (u,v), point_size, 255, -1)
    return image

image_size = (2048, 512)
img1 = project_points_with_P(curve_3d, P1, image_size)
img2 = project_points_with_P(curve_3d, P2, image_size)
cv2.imwrite('img1.png', img1)
cv2.imwrite('img2.png', img2)