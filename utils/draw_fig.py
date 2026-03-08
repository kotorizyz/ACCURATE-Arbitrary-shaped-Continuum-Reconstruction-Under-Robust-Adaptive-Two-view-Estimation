import numpy as np
import matplotlib.pyplot as plt
import cv2
import open3d as o3d
import argparse

def plot_pointcloud_two_views(P, save_prefix=None, cate = 'gt', axis_len_x = None, axis_len_y = None, axis_len_z = None):
    assert P.shape[1] == 3

    xmin, ymin, zmin = P.min(axis=0)
    xmax, ymax, zmax = P.max(axis=0)

    margin = 0.08 * max(xmax-xmin, ymax-ymin, zmax-zmin)
    xmin -= margin; ymin -= margin; zmin -= margin
    xmax += margin; ymax += margin; zmax += margin

    if axis_len_x == None:
        axis_len_x = xmax-xmin * 1
        axis_len_y = ymax-ymin * 1
        axis_len_z = zmax-zmin * 0.98

    views = [(20, 40), (10, -60)]
    # views = [(30, -60), (30, 120)]

    color_point = "#FC4F7E"

    for idx, (elev, azim) in enumerate(views):
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(P[:,0], P[:,1], P[:,2],
                   s=4, c=color_point, alpha=0.95)

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
        

        # grid_color = '#E6E6E6'
        grid_color = '#000000'
        grid_alpha = 0.6
        grid_step = (axis_len_x + axis_len_y + axis_len_z) / 30
        color_xy = '#A7C7E7'   # very light blue
        color_yz = '#CDECCF'   # very light green
        color_zx = '#FFE1B5'   # very light orange
        plane_alpha = 0.3

        xx = np.arange(origin[0], origin[0] + axis_len_x, grid_step)
        yy = np.arange(origin[1], origin[1] + axis_len_y, grid_step)
        XX, YY = np.meshgrid(xx, yy)
        ZZ = np.full_like(XX, origin[2])
        ax.plot_surface(
            XX, YY, ZZ,
            color=color_xy,
            alpha=plane_alpha,
            linewidth=0,
            antialiased=True,
            shade=False
        )
        ax.plot_wireframe(XX, YY, ZZ, color=grid_color, alpha=grid_alpha, linewidth=0.5)

        yy = np.arange(origin[1], origin[1] + axis_len_y, grid_step)
        zz = np.arange(origin[2], origin[2] + axis_len_z, grid_step)
        YY, ZZ = np.meshgrid(yy, zz)
        XX = np.full_like(YY, origin[0])
        ax.plot_surface(
            XX, YY, ZZ,
            color=color_yz,
            alpha=plane_alpha,
            linewidth=0,
            antialiased=True,
            shade=False
        )
        ax.plot_wireframe(XX, YY, ZZ, color=grid_color, alpha=grid_alpha, linewidth=0.5)

        zz = np.arange(origin[2], origin[2] + axis_len_z, grid_step)
        xx = np.arange(origin[0], origin[0] + axis_len_x, grid_step)
        ZZ, XX = np.meshgrid(zz, xx)
        YY = np.full_like(ZZ, origin[1])
        ax.plot_surface(
            XX, YY, ZZ,
            color=color_zx,
            alpha=plane_alpha,
            linewidth=0,
            antialiased=True,
            shade=False
        )
        ax.plot_wireframe(XX, YY, ZZ, color=grid_color, alpha=grid_alpha, linewidth=0.5)

        ax.text(origin[0]+axis_len_x*1.2, origin[1], origin[2], 'X', color='#000000', fontsize=12)
        ax.text(origin[0], origin[1]+axis_len_y*1.1, origin[2], 'Y', color='#000000', fontsize=12)
        ax.text(origin[0], origin[1], origin[2]+axis_len_z*0.99, 'Z', color='#000000', fontsize=12)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)

        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.grid(False)
        ax.set_axis_off()
        ax.set_box_aspect([xmax-xmin, ymax-ymin, zmax-zmin])

        ax.view_init(elev=elev, azim=azim)

        if save_prefix:
            plt.savefig(f"{save_prefix}_{cate}_view{idx+1}.png",
                        dpi=300, bbox_inches='tight', transparent=True)
        
        return axis_len_x, axis_len_y, axis_len_z

def compute_shared_axes(P_ref, margin_ratio=0.08):
    xmin, ymin, zmin = P_ref.min(axis=0)
    xmax, ymax, zmax = P_ref.max(axis=0)

    margin = margin_ratio * max(xmax-xmin, ymax-ymin, zmax-zmin)
    xmin -= margin; ymin -= margin; zmin -= margin
    xmax += margin; ymax += margin; zmax += margin

    axis_len_x = xmax - xmin
    axis_len_y = ymax - ymin
    axis_len_z = zmax - zmin * 0.98

    origin = np.array([xmin, ymin, zmin])

    return {
        'xmin': xmin, 'xmax': xmax,
        'ymin': ymin, 'ymax': ymax,
        'zmin': zmin, 'zmax': zmax,
        'axis_len_x': axis_len_x,
        'axis_len_y': axis_len_y,
        'axis_len_z': axis_len_z,
        'origin': origin
    }

def plot_pointcloud_with_axes(P, axes_cfg, views, color, save_path=None):
    xmin = axes_cfg['xmin']; xmax = axes_cfg['xmax']
    ymin = axes_cfg['ymin']; ymax = axes_cfg['ymax']
    zmin = axes_cfg['zmin']; zmax = axes_cfg['zmax']
    origin = axes_cfg['origin']
    axis_len_x = axes_cfg['axis_len_x']
    axis_len_y = axes_cfg['axis_len_y']
    axis_len_z = axes_cfg['axis_len_z']

    for i, (elev, azim) in enumerate(views):
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(P[:,0], P[:,1], P[:,2],
                   s=4, c=color, alpha=0.95)

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
        

        # grid_color = '#E6E6E6'
        grid_color = '#000000'
        grid_alpha = 0.6
        grid_step = (axis_len_x + axis_len_y + axis_len_z) / 30
        color_xy = '#A7C7E7'   # very light blue
        color_yz = '#CDECCF'   # very light green
        color_zx = '#FFE1B5'   # very light orange
        plane_alpha = 0.3

        xx = np.arange(origin[0], origin[0] + axis_len_x, grid_step)
        yy = np.arange(origin[1], origin[1] + axis_len_y, grid_step)
        XX, YY = np.meshgrid(xx, yy)
        ZZ = np.full_like(XX, origin[2])
        ax.plot_surface(
            XX, YY, ZZ,
            color=color_xy,
            alpha=plane_alpha,
            linewidth=0,
            antialiased=True,
            shade=False
        )
        ax.plot_wireframe(XX, YY, ZZ, color=grid_color, alpha=grid_alpha, linewidth=0.5)

        yy = np.arange(origin[1], origin[1] + axis_len_y, grid_step)
        zz = np.arange(origin[2], origin[2] + axis_len_z, grid_step)
        YY, ZZ = np.meshgrid(yy, zz)
        XX = np.full_like(YY, origin[0])
        ax.plot_surface(
            XX, YY, ZZ,
            color=color_yz,
            alpha=plane_alpha,
            linewidth=0,
            antialiased=True,
            shade=False
        )
        ax.plot_wireframe(XX, YY, ZZ, color=grid_color, alpha=grid_alpha, linewidth=0.5)

        zz = np.arange(origin[2], origin[2] + axis_len_z, grid_step)
        xx = np.arange(origin[0], origin[0] + axis_len_x, grid_step)
        ZZ, XX = np.meshgrid(zz, xx)
        YY = np.full_like(ZZ, origin[1])
        ax.plot_surface(
            XX, YY, ZZ,
            color=color_zx,
            alpha=plane_alpha,
            linewidth=0,
            antialiased=True,
            shade=False
        )
        ax.plot_wireframe(XX, YY, ZZ, color=grid_color, alpha=grid_alpha, linewidth=0.5)

        ax.text(origin[0]+axis_len_x*1.2, origin[1], origin[2], 'X', color='#000000', fontsize=12)
        ax.text(origin[0], origin[1]+axis_len_y*1.1, origin[2], 'Y', color='#000000', fontsize=12)
        ax.text(origin[0], origin[1], origin[2]+axis_len_z*0.99, 'Z', color='#000000', fontsize=12)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)

        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.grid(False)
        ax.set_axis_off()
        ax.set_box_aspect([xmax-xmin, ymax-ymin, zmax-zmin])

        ax.view_init(elev=elev, azim=azim)
        if save_path:
            plt.savefig(f"{save_path}_view{i+1}.png",
                        dpi=300, bbox_inches='tight', transparent=True)

def inverse_img(img):
    img_f = np.array(img, dtype=np.float32)
    img_inv = np.abs(img_f - 255)
    return img_inv


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--category',
        type=str,
        default='simulation',
        help='Dataset category: simulation or phantom (default: simulation)'
    )

    parser.add_argument(
        '--receive',
        type=str,
        default='mask',
        help='Input type: image, mask or point (default: mask)'
    )

    parser.add_argument(
        '--method',
        type=str,
        default='ACCURATE',
        help='Reconstruction method: ACCURATE/TMI03/TMI15/vggt/Fast3R/MonSter (default: ACCURATE)'
    )
    args = parser.parse_args()


    category = args.category
    method = args.method
    receive = args.receive

    if category == 'phantom':
        path_rec = f'experiment/{receive}/{category}/{method}/case_151.ply'
        path_gt = f'ACCURATE_dataset/{category}/case_151/annotations/guidewire_3D.ply'
    else:
        path_rec = f'experiment/{receive}/{category}/{method}/case_003.ply'
        path_gt = f'ACCURATE_dataset/{category}/case_003/annotations/guidewire_3D.ply'

    # P = np.load('pts.npy')
    # P = o3d.io.read_point_cloud(f'ctdataset/SE6/Segment_0.ply')
    # P = o3d.io.read_point_cloud('output/rec_0.ply')
    P_gt = o3d.io.read_point_cloud(path_gt)
    P_gt = np.asarray(P_gt.points)  # (N, 3)
    P_rec = o3d.io.read_point_cloud(path_rec)
    P_rec = np.asarray(P_rec.points)  # (N, 3)

    if category == 'phantom':
        P_gt = P_gt * [[-1,-1,1]]
        P_rec = P_rec * [[-1,-1,1]]

    axes_cfg = compute_shared_axes(P_gt)
    views = [(20, 40), (10, -60)]

    # views = [(20, -100), (10, -60)]

    plot_pointcloud_with_axes(
        P_gt, axes_cfg, views,
        color='#1F77B4',   # blue
        save_path='gt'
    )

    plot_pointcloud_with_axes(
        P_rec, axes_cfg, views,
        color='#FC4F7E',   # pink
        save_path='rec'
    )

    # mask_L = cv2.imread('mask/mask_L_0.png', cv2.IMREAD_GRAYSCALE)
    # mask_R = cv2.imread('mask/mask_R_0.png', cv2.IMREAD_GRAYSCALE)

    # mask_L = inverse_img(mask_L)
    # mask_R = inverse_img(mask_R)

    # cv2.imwrite('mask_L_inverted.png', mask_L)
    # cv2.imwrite('mask_R_inverted.png', mask_R)