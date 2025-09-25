import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.morphology import skeletonize
from skan import Skeleton, summarize
from collections import defaultdict

def ordered_points_from_mask(mask):

    def longest_path_with_cycles(edges):
        g = defaultdict(list)
        for idx, (u, v, l) in enumerate(edges):
            g[u].append((v, l, idx))
            g[v].append((u, l, idx))

        longest_path = []
        max_length = 0

        def dfs(node, used_edges, path, length):
            nonlocal longest_path, max_length
            if length > max_length:
                max_length = length
                longest_path = path[:]
            for nei, w, idx in g[node]:
                if idx not in used_edges:
                    used_edges.add(idx)
                    path.append(nei)
                    dfs(nei, used_edges, path, length + w)
                    path.pop()
                    used_edges.remove(idx)

        for start in g:
            dfs(start, set(), [start], 0)

        return longest_path, max_length

    # mask = io.imread('sim_data/mask/maskthick_cam2.png') > 10  # 二值化
    # mask_skel = skeletonize(mask)
    skeleton = Skeleton(mask)
    summary_df = summarize(skeleton)

    path_ids = skeleton.paths_list()

    paths = []
    for i, row in summary_df.iterrows():
        paths.append((row['node-id-src'], row['node-id-dst'], row['branch-distance']))

    path_nodes, total_length = longest_path_with_cycles(paths)
    # print("Longest path node IDs:", path_nodes)
    # print("Total length:", total_length)

    paths_used = [False] * len(paths)
    guidewire_ids = []
    for i in range(len(path_nodes)-1):
        start_node = path_nodes[i]
        end_node = path_nodes[i+1]
        for path in paths:
            if ((path[0] == start_node and path[1] == end_node)):
                idx = paths.index(path)
                if(paths_used[idx]):
                    continue
                paths_used[idx] = True
                guidewire_ids += path_ids[idx]
                # print(path_ids[idx][0], path_ids[idx][-1], len(path_ids[idx]))
                break
            elif ((path[1] == start_node and path[0] == end_node)):
                idx = paths.index(path)
                if(paths_used[idx]):
                    continue
                paths_used[idx] = True
                guidewire_ids += path_ids[idx][::-1]
                # print(path_ids[idx][-1], path_ids[idx][0], len(path_ids[idx]))
                break
    # print(len(guidewire_ids))

    ordered_points = [skeleton.coordinates[idx] for idx in guidewire_ids]
    ordered_points = np.array(ordered_points).astype(int)
    # ordered_points = ordered_points

    # img = np.zeros(mask.shape)
    # for p in ordered_points:
    #     y, x = p
    #     img[y, x] = 1
    # plt.imsave('guidewire_cam1_ordered.png', img, cmap='gray')

    return ordered_points