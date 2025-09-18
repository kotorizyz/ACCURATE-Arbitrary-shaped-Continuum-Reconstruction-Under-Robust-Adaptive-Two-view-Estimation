import numpy as np
from collections import defaultdict, deque
from skimage.morphology import skeletonize
import cv2

def get_neighbors(coord, mask):
    x, y = coord
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < mask.shape[0] and 0 <= ny < mask.shape[1]:
                if mask[nx, ny] == 1:
                    neighbors.append((nx, ny))
    return neighbors

def order_mask_points(mask):
    points = set(map(tuple, np.argwhere(mask == 1)))
    
    graph = defaultdict(list)
    for pt in points:
        for nb in get_neighbors(pt, mask):
            graph[pt].append(nb)
    
    endpoints = [pt for pt, nbs in graph.items() if len(nbs) == 1]
    if not endpoints:
        raise ValueError("没有找到端点，可能是闭合曲线")
    
    start = endpoints[0]
    
    ordered = []
    visited = set()
    stack = deque([start])
    
    prev = None
    while stack:
        pt = stack.pop()
        if pt in visited:
            continue
        visited.add(pt)
        ordered.append(pt)
        for nb in graph[pt]:
            if nb not in visited:
                stack.append(nb)
    return ordered

mask = cv2.imread("mask/maskthick_cam2.png", cv2.IMREAD_GRAYSCALE)
skeleton = skeletonize(mask > 10).astype(np.uint8)
# cv2.imwrite("mask.png", skeleton * 255)
coords = order_mask_points(mask)
print(coords)