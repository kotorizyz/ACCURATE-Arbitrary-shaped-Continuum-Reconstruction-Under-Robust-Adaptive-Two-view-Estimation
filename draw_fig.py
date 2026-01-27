import numpy as np
import matplotlib.pyplot as plt
import cv2
import open3d as o3d

def plot_pointcloud_two_views(P, save_prefix=None):
    assert P.shape[1] == 3

    xmin, ymin, zmin = P.min(axis=0)
    xmax, ymax, zmax = P.max(axis=0)

    margin = 0.08 * max(xmax-xmin, ymax-ymin, zmax-zmin)
    xmin -= margin; ymin -= margin; zmin -= margin
    xmax += margin; ymax += margin; zmax += margin

    axis_len_x = xmax-xmin * 1
    axis_len_y = ymax-ymin * 1
    axis_len_z = zmax-zmin * 0.98

    views = [(20, 40), (10, -60)]
    # views = [(30, -60), (30, 120)]


    # 🎨 配色
    color_point = "#FFFFFF"

    for idx, (elev, azim) in enumerate(views):
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111, projection='3d')

        # # 点云
        # ax.scatter(P[:,0], P[:,1], P[:,2],
        #            s=4, c=color_point, alpha=0.95)

        # 三轴
        origin = np.array([xmin, ymin, zmin])
        ax.quiver(origin[0], origin[1], origin[2],
          axis_len_x, 0, 0,
          color='#000000', linewidth=2.5, arrow_length_ratio=0.08)

        ax.quiver(origin[0], origin[1], origin[2],
                0, axis_len_y, 0,
                color='#000000', linewidth=2.5, arrow_length_ratio=0.08)

        ax.quiver(origin[0], origin[1], origin[2],
                0, 0, axis_len_z,
                color='#000000', linewidth=2.5, arrow_length_ratio=0.08)
        

        # grid_color = '#E6E6E6'   # 很淡的灰色
        grid_color = '#000000'
        grid_alpha = 0.6
        grid_step = (axis_len_x + axis_len_y + axis_len_z) / 30  # 自动网格密度

        # XY 平面 (Z = origin[2])
        xx = np.arange(origin[0], origin[0] + axis_len_x, grid_step)
        yy = np.arange(origin[1], origin[1] + axis_len_y, grid_step)
        XX, YY = np.meshgrid(xx, yy)
        ZZ = np.full_like(XX, origin[2])
        ax.plot_wireframe(XX, YY, ZZ, color=grid_color, alpha=grid_alpha, linewidth=0.5)

        # YZ 平面 (X = origin[0])
        yy = np.arange(origin[1], origin[1] + axis_len_y, grid_step)
        zz = np.arange(origin[2], origin[2] + axis_len_z, grid_step)
        YY, ZZ = np.meshgrid(yy, zz)
        XX = np.full_like(YY, origin[0])
        ax.plot_wireframe(XX, YY, ZZ, color=grid_color, alpha=grid_alpha, linewidth=0.5)

        # ZX 平面 (Y = origin[1])
        zz = np.arange(origin[2], origin[2] + axis_len_z, grid_step)
        xx = np.arange(origin[0], origin[0] + axis_len_x, grid_step)
        ZZ, XX = np.meshgrid(zz, xx)
        YY = np.full_like(ZZ, origin[1])
        ax.plot_wireframe(XX, YY, ZZ, color=grid_color, alpha=grid_alpha, linewidth=0.5)

        # 轴标签
        ax.text(origin[0]+axis_len_x*1.2, origin[1], origin[2], 'X', color='#000000', fontsize=12)
        ax.text(origin[0], origin[1]+axis_len_y*1.1, origin[2], 'Y', color='#000000', fontsize=12)
        ax.text(origin[0], origin[1], origin[2]+axis_len_z*1.04, 'Z', color='#000000', fontsize=12)

        # 范围
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)

        # 极简外观
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.grid(False)
        ax.set_axis_off()
        ax.set_box_aspect([xmax-xmin, ymax-ymin, zmax-zmin])

        # 视角
        ax.view_init(elev=elev, azim=azim)

        if save_prefix:
            plt.savefig(f"{save_prefix}_view{idx+1}.png",
                        dpi=300, bbox_inches='tight', transparent=True)

def inverse_img(img):
    img_f = np.array(img, dtype=np.float32)
    img_inv = np.abs(img_f - 255)
    return img_inv

P = np.load('pts.npy')
# P = o3d.io.read_point_cloud(f'ctdataset/SE6/Segment_0.ply')
P = o3d.io.read_point_cloud('output/rec_0.ply')
P = np.asarray(P.points)  # (N, 3)
plot_pointcloud_two_views(P, save_prefix='pointcloud')
quit()

# mask_L = cv2.imread('mask/mask_L_0.png', cv2.IMREAD_GRAYSCALE)
# mask_R = cv2.imread('mask/mask_R_0.png', cv2.IMREAD_GRAYSCALE)

# mask_L = inverse_img(mask_L)
# mask_R = inverse_img(mask_R)

# cv2.imwrite('mask_L_inverted.png', mask_L)
# cv2.imwrite('mask_R_inverted.png', mask_R)