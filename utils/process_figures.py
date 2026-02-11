import numpy as np
import cv2
from skimage.morphology import skeletonize
from skimage.draw import line

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

def thicken_skeleton(skel, thickness=3):
    """
    skel: 2D binary skeleton image (0 or 255 / True or False)
    thickness: 线宽 (推荐 2~6)
    """

    # 确保是uint8二值
    skel_bin = (skel > 0).astype(np.uint8) * 255

    # 构造圆形结构元素（比方形更自然）
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, 
        (thickness, thickness)
    )

    thick = cv2.dilate(skel_bin, kernel)

    return thick

if __name__ == "__main__":
    mask = np.load('tmp/data0_gt_168.npz')
    mask = (mask['probabilities'][1,0] > 0.5) * 1.0
    mask = fix_single_pixel_gaps(mask, 10)
    mask_thick = thicken_skeleton(mask)
    cv2.imwrite('show.png', mask_thick)