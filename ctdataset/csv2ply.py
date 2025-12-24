import open3d as o3d
import numpy as np
import csv

# -----------------------------
# 1. 从 CSV 读取点
# -----------------------------
csv_path = 'seg_pointcloud.csv'  # CSV 路径
points = []

with open(csv_path, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # 跳过表头
    for row in reader:
        x, y, z = map(float, row)
        points.append([x, y, z])

points = np.array(points)

# -----------------------------
# 2. 创建点云对象
# -----------------------------
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# -----------------------------
# 3. 保存为 PLY
# -----------------------------
ply_path = 'seg_pointcloud.ply'
o3d.io.write_point_cloud(ply_path, pcd)

print(f"点云已保存为 {ply_path}，共 {len(points)} 个点")
