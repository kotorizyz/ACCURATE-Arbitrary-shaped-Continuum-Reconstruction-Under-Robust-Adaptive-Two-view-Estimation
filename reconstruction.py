import cv2
import numpy as np
import json
import open3d as o3d

cnt = np.zeros((3000, 4096))

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

    # 3d points
    pts_3d = []

    # No mask points in right image
    if(len(np.where(mask2==1)[0])==0):
        return np.array(pts_3d)

    # Threshold for intersection distance
    col = np.sqrt(2)/2

    for (u_l, v_l) in polyline_left:
        line_right = cv2.computeCorrespondEpilines(np.array([[[u_l, v_l]]], dtype=np.float32), 1, F).reshape(-1, 3)[0]
        a, b, c = line_right
        # img = mask2
        # img = draw_epiline(img, line_right)
        # cv2.imwrite(f"epiline_{u_l}_{v_l}.png", img*255)
        # quit()
        ys, xs = np.where(mask2 == 1)
        dist = np.abs(a*xs + b*ys + c) / np.sqrt(a*a + b*b)
        pts_dist = list(zip(xs, ys, dist))
        pts_dist.sort(key=lambda x: x[2])
        pts_dist = pts_dist[:2]
        if(pts_dist[0][2] > col):
            continue
        sum_dist = pts_dist[0][2] + pts_dist[1][2] + 1e-5
        # print(pts_dist)
        x_weighted = pts_dist[0][0] * (pts_dist[1][2] / sum_dist) + pts_dist[1][0] * (pts_dist[0][2] / sum_dist)
        y_weighted = pts_dist[0][1] * (pts_dist[1][2] / sum_dist) + pts_dist[1][1] * (pts_dist[0][2] / sum_dist)
        # print(pts_dist, (x_weighted, y_weighted))
        pt_l = np.array([u_l, v_l], dtype=np.float32).reshape(2, 1)
        pt_r = np.array([x_weighted, y_weighted], dtype=np.float32).reshape(2, 1)
        pt_l_ud = cv2.undistortPoints(np.expand_dims(pt_l, axis=1), mtx_l, dist_l)
        pt_r_ud = cv2.undistortPoints(np.expand_dims(pt_r, axis=1), mtx_r, dist_r)
        points4D = cv2.triangulatePoints(P1, P2, pt_l_ud.T, pt_r_ud.T)
        pt_3d = (points4D[:3] / points4D[3]).flatten()  # (x,y,z)
        print(pt_3d)
        # pts_4d, err = triangulate_closest(P1, P2, pt_l, pt_r)
        # pt_3d = (pts_4d[:3] / pts_4d[3]).ravel()
        pts_3d.append(pt_3d)
        # print(pt_3d)
    return np.array(pts_3d)


params = np.load("stereo_params.npz")
mtx_l, dist_l = params["mtx_l"], params["dist_l"]
mtx_r, dist_r = params["mtx_r"], params["dist_r"]
R, T = params["R"], params["T"]
F = params["F"]
P1 = mtx_l @ np.hstack((np.eye(3), np.zeros((3,1))))
P2 = mtx_r @ np.hstack((R, T))

mask1, start_point1, end_point1 = json_to_mask("rec_image/C1.json", (3000, 4096))
mask2, start_point2, end_point2 = json_to_mask("rec_image/C2.json", (3000, 4096))

polyline_left = np.array(np.where(mask1 == 1)).T[:, ::-1]
polyline_left = polyline_left[np.argsort(polyline_left[:, 0])]
curve_3d = reconstruct_polyline_3d(polyline_left, mask2, P1, P2, F)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(curve_3d)
o3d.io.write_point_cloud("guidewire.ply", pcd)