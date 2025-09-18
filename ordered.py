import numpy as np
from collections import defaultdict, deque
from skimage.morphology import skeletonize
import cv2

import numpy as np
from skimage.morphology import skeletonize
from collections import defaultdict, deque

def get_neighbors(coord, mask):
    x, y = coord
    for dx in [-1,0,1]:
        for dy in [-1,0,1]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = x+dx, y+dy
            if 0 <= nx < mask.shape[0] and 0 <= ny < mask.shape[1]:
                if mask[nx,ny] == 1:
                    yield (nx,ny)

def build_graph(mask):
    graph = defaultdict(list)
    coords = np.argwhere(mask==1)
    for x,y in map(tuple, coords):
        for nb in get_neighbors((x,y), mask):
            graph[(x,y)].append(nb)
    return graph

def find_endpoints(graph):
    return [pt for pt,nbs in graph.items() if len(nbs)==1]

def bfs_path(graph, start, end):
    queue = deque([(start,[start])])
    visited = set([start])
    while queue:
        node, path = queue.popleft()
        if node == end:
            return path
        for nb in graph[node]:
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, path+[nb]))
    return None

def longest_path(mask):
    graph = build_graph(mask)
    endpoints = find_endpoints(graph)
    print(endpoints)
    if len(endpoints)<2:
        raise ValueError("未找到足够的端点")
    max_len=0
    best_path=[]
    for i in range(len(endpoints)):
        for j in range(i+1,len(endpoints)):
            p=bfs_path(graph,endpoints[i],endpoints[j])
            if p and len(p)>max_len:
                max_len=len(p)
                best_path=p
    return best_path

mask = cv2.imread("sim_data/mask/maskthick_cam1.png", cv2.IMREAD_GRAYSCALE)
skeleton = skeletonize(mask > 10).astype(np.uint8)
# cv2.imwrite("mask.png", skeleton * 255)
coords = longest_path(mask)
print(coords)