import numpy as np
import cv2
import glob

chessboard_size = (8, 11)
square_size = 5

objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []
imgpoints_left = [] 
imgpoints_right = []

images_left = sorted(glob.glob("calibration/C1*.bmp"))
images_right = sorted(glob.glob("calibration/C2*.bmp"))

images_left_eff = []
images_right_eff = []

assert len(images_left) == len(images_right)

for left_path, right_path in zip(images_left, images_right):
    img_left = cv2.imread(left_path)
    img_right = cv2.imread(right_path)
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    ret_left, corners_left = cv2.findChessboardCornersSB(gray_left, chessboard_size, None)
    ret_right, corners_right = cv2.findChessboardCornersSB(gray_right, chessboard_size, None)

    print(ret_left, ret_right)
    if ret_left and ret_right:
        images_left_eff.append(left_path)
        images_right_eff.append(right_path)

        objpoints.append(objp)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)    #(88, 1, 2)
        corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria) #(88, 1, 2)

        imgpoints_left.append(corners_left)
        imgpoints_right.append(corners_right)

        cv2.drawChessboardCorners(img_left, chessboard_size, corners_left, ret_left)
        cv2.drawChessboardCorners(img_right, chessboard_size, corners_right, ret_right)
        cv2.imwrite("left.png", img_left)
        cv2.imwrite("right.png", img_right)

# 单目标定
ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(objpoints, imgpoints_left, gray_left.shape[::-1], None, None)
ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(objpoints, imgpoints_right, gray_right.shape[::-1], None, None)

# 双目标定
flags = cv2.CALIB_FIX_INTRINSIC  # 固定内参，只优化外参
criteria_stereo = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right,
    mtx_l, dist_l, mtx_r, dist_r,
    gray_left.shape[::-1], criteria=criteria_stereo, flags=flags
)
print("旋转矩阵 R:\n", R)
print("平移向量 T:\n", T)

print(dist_l)
print(dist_r)

def reproj_errors(objpoints, imgpoints, rvecs, tvecs, K, dist):
    errors = []
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        err = cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / len(proj)
        errors.append(err)
    return errors

errs = reproj_errors(objpoints, imgpoints_left, rvecs_l, tvecs_l, mtx_l, dist_l)
for i,e in enumerate(errs): print(i, e)

R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    mtx_l, dist_l, mtx_r, dist_r,
    gray_left.shape[::-1], R, T, alpha=0
)

np.savez("stereo_params.npz",
         mtx_l=mtx_l, dist_l=dist_l,
         mtx_r=mtx_r, dist_r=dist_r,
         R=R, T=T, E=E, F=F,
         R1=R1, R2=R2, P1=P1, P2=P2, Q=Q)
print("标定参数已保存到 stereo_params.npz")