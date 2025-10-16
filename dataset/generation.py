import bpy
import bmesh
import mathutils
from mathutils import Vector, Matrix
import math
import random
import numpy as np
import json
import os
from pathlib import Path
def create_guidewire():

    # 随机总长度
    main_length = random.uniform(50, 120)
    j_length = 3.0
    
    # 创建曲线数据
    curve_data = bpy.data.curves.new('guidewire', type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.resolution_u = 12
    
    spline = curve_data.splines.new('BEZIER')
    
    # 控制点数量
    if main_length < 80:
        main_points = 10
    else:
        main_points = int(main_length / 10) + 10
    
    # J型部分点数（不包括连接点）
    j_points = 5  
    total_points = main_points + j_points  # 总点数 = 主体点数 + J型点数
    
    spline.bezier_points.add(total_points - 1)
    points = spline.bezier_points
    
    print(f"控制点分布: 主体 {main_points} 点, J型 {j_points} 点, 总计 {total_points} 点")
    
    points[0].co = Vector((0, 0, 0))
    points[0].handle_left_type = 'AUTO'
    points[0].handle_right_type = 'AUTO'
    # 生成主体部分（0 到 main_points-1）
    for i in range(1,main_points):
        if main_length < 80:
            # 较短导丝，整体平缓
            x = random.uniform(-5, 5)
            y = random.uniform(-5, 5)
            z_variation = random.uniform(-5, 5)
            base_z = main_length / (main_points - 1) * i
            points[i].co = Vector((x, y, base_z + z_variation))
        else:
            t = i / (main_points - 1)
            parametric_t = t ** 0.7
            base_z = parametric_t * main_length
            
            # 波动设置
            if i <= 3:
                x = random.uniform(-3, 3)
                y = random.uniform(-3, 3)
                z_variation = random.uniform(-2, 2)
            elif i >= main_points - 4:
                # 主体末端，为J型做准备，减小波动
                x = random.uniform(-4, 4)
                y = random.uniform(-3, 3)
                z_variation = random.uniform(-2, 2)
            else:
                distance_factor = parametric_t
                max_variation = 18 * distance_factor
                x = random.uniform(-max_variation, max_variation)
                y = random.uniform(-max_variation, max_variation)
                
                if parametric_t > 0.3 and parametric_t < 0.7 and random.random() > 0.85:
                    z_variation = random.uniform(-max_variation, -max_variation * 0.2)
                else:
                    z_variation = random.uniform(-max_variation * 0.7, max_variation * 0.7)
        points[i].co = Vector((x, y, base_z + z_variation))
        points[i].handle_left_type = 'AUTO'
        points[i].handle_right_type = 'AUTO'
    
    # 获取主体末端方向（用于J型起始方向）
    if main_points >= 2:
        end_direction = (points[main_points-1].co - points[main_points-2].co).normalized()
    else:
        end_direction = Vector((0, 0, 1))
    
    # 选择弯曲方向
    bend_direction = Vector((1, 0, 0)) if random.random() > 0.5 else Vector((-1, 0, 0))
     
    for j in range(j_points):
        i = main_points + j  # J型点的索引
        t = (j + 1) / (j_points + 1)  # 参数化，从1/6到5/6避免边界问题
        
        # 平滑的弯曲函数
        bend_factor = t * t  # 二次函数，起始平滑
        
        # 计算J型点位置
        # 起点是主体末端 points[main_points-1].co
        forward = end_direction * j_length * t * 0.6
        side_bend = bend_direction * j_length * bend_factor * 0.8
        downward = Vector((0, 0, -j_length * bend_factor * 0.2))
        
        # 从主体末端开始计算偏移
        base_position = points[main_points-1].co
        points[i].co = base_position + forward + side_bend + downward
        points[i].handle_left_type = 'AUTO'
        points[i].handle_right_type = 'AUTO'
    
    # 特别检查连接处（主体末端和J型第一个点）
    if main_points >= 1 and j_points >= 1:
        connection_point = points[main_points-1]  # 主体末端
        j_first_point = points[main_points]       # J型第一个点       
        # 确保两点不重合
        if (j_first_point.co - connection_point.co).length < 0.1:
            # 轻微调整J型第一点位置
            adjustment = end_direction * 0.5
            j_first_point.co += adjustment
    
    # 创建曲线对象
    curve_obj = bpy.data.objects.new('Guidewire', curve_data)
    bpy.context.collection.objects.link(curve_obj)
    
    # 计算信息
    bpy.context.view_layer.update()

    return {
        'curve_object': curve_obj
    }
def get_camera_params(camera_obj):
    f_in_mm = camera_obj.data.lens
    scene = bpy.context.scene

    #考虑渲染分辨率比例
    resolution_percentage = scene.render.resolution_percentage / 100.0
    resolution_x = int(scene.render.resolution_x * resolution_percentage)
    resolution_y = int(scene.render.resolution_y * resolution_percentage)

    # 获取传感器尺寸
    sensor_width_mm = camera_obj.data.sensor_width
    sensor_height_mm = camera_obj.data.sensor_height
    
    # 计算输出比例和传感器比例
    output_aspect = resolution_x / resolution_y
    sensor_aspect = sensor_width_mm / sensor_height_mm
    
    # 根据sensor_fit模式计算有效传感器尺寸
    sensor_fit = camera_obj.data.sensor_fit
    
    if sensor_fit == 'AUTO':
        # AUTO模式：根据输出比例自动选择
        if output_aspect > sensor_aspect:
            # 输出更宽，使用水平适配
            effective_sensor_width = sensor_width_mm
            effective_sensor_height = sensor_width_mm / output_aspect
        else:
            # 输出更高，使用垂直适配
            effective_sensor_height = sensor_height_mm
            effective_sensor_width = sensor_height_mm * output_aspect
    elif sensor_fit == 'HORIZONTAL':
        # 水平适配：保持传感器宽度，调整高度
        effective_sensor_width = sensor_width_mm
        effective_sensor_height = sensor_width_mm / output_aspect
    else:  # 'VERTICAL'
        # 垂直适配：保持传感器高度，调整宽度
        effective_sensor_height = sensor_height_mm
        effective_sensor_width = sensor_height_mm * output_aspect
    
    # 计算像素尺寸和焦距
    pixel_size_x = effective_sensor_width / resolution_x
    pixel_size_y = effective_sensor_height / resolution_y
    
    fx = f_in_mm / pixel_size_x
    fy = f_in_mm / pixel_size_y
    
    cx = resolution_x / 2
    cy = resolution_y / 2
    
    intrinsic = np.array((
        (fx, 0, cx),
        (0, fy, cy),
        (0, 0, 1)
    ))
    matrix_world = camera_obj.matrix_world.inverted()
    matrix_np = np.array(matrix_world)
    transform = np.array([[1, 0, 0, 0],
                          [0, -1, 0, 0],
                          [0, 0, -1, 0],
                          [0, 0, 0, 1]])
    extrinsic = np.dot(transform, matrix_np)

    return {
            'intrinsic': intrinsic,
            'extrinsic': extrinsic,
            'resolution': (resolution_x, resolution_y)
    }
def setup_cameras(curve_obj):
    """
    设置两个相机：一个拍摄xoz平面，另一个从xy轴中间部分向y轴看过去的角度
    返回两个相机的内参和外参
    """
    # 获取曲线边界框以确定相机位置
    bbox = [curve_obj.matrix_world @ Vector(corner) for corner in curve_obj.bound_box]
    bbox_center = sum(bbox, Vector()) / 8
    
    # 计算边界框大小
    bbox_min = Vector((min(v.x for v in bbox), min(v.y for v in bbox), min(v.z for v in bbox)))
    bbox_max = Vector((max(v.x for v in bbox), max(v.y for v in bbox), max(v.z for v in bbox)))
    bbox_size = bbox_max - bbox_min
    
    # 计算曲线长度
    curve_length = max(bbox_size.x, bbox_size.y, bbox_size.z)
    
    # 相机1：拍摄xoz平面
    cam1 = bpy.data.cameras.new("Camera_XOZ")
    cam1_obj = bpy.data.objects.new("Camera_XOZ", cam1)
    bpy.context.collection.objects.link(cam1_obj)
    
    # 设置相机1位置和方向
    cam_distance = curve_length * 2.0  # 增加距离确保整个曲线可见
    cam1_obj.location = Vector((cam_distance, 0, bbox_center.z))
    cam1_obj.rotation_euler = (math.pi/2, 0, math.pi/2)  # 朝向xoz平面

    # 调整相机视野以确保整个曲线可见
    cam1.lens = 35
    cam1_obj.data.clip_end = 500
    cam1_obj.data.shift_x = 0.0
    cam1_obj.data.shift_y = 0.0

    bpy.context.view_layer.update()  #强制更新矩阵计算

    # 相机2：从xy轴中间部分向y轴看过去的角度
    cam2 = bpy.data.cameras.new("Camera_XY")
    cam2_obj = bpy.data.objects.new("Camera_XY", cam2)
    bpy.context.collection.objects.link(cam2_obj)
    
    # 随机角度（30-50度）
    angle_deg = random.uniform(30, 50)
    angle_rad = math.radians(angle_deg)
    
    # 计算相机2位置
    x_pos = cam_distance * math.cos(angle_rad)
    y_pos = cam_distance * math.sin(angle_rad)
    cam2_obj.location = Vector((x_pos, y_pos, bbox_center.z))
    
    # 看向曲线中心
    direction = (bbox_center - cam2_obj.location).normalized()
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam2_obj.rotation_euler = rot_quat.to_euler()

    # 调整相机2视野
    cam2.lens = 35
    cam2_obj.data.clip_end = 500
    cam2_obj.data.shift_x = 0.0
    cam2_obj.data.shift_y = 0.0   
    bpy.context.view_layer.update()  # 关键：强制更新矩阵计算   
      
    # 计算相机内参和外参
    cam1_params = get_camera_params(cam1_obj)
    cam2_params = get_camera_params(cam2_obj)
    
    return {
        'cam1': {'object': cam1_obj, 'params': cam1_params},
        'cam2': {'object': cam2_obj, 'params': cam2_params}
    }
def setup_mask_materials():
    """设置曲线自发光材质"""
    curve_mat = bpy.data.materials.new(name="Curve_Emission")
    curve_mat.use_nodes = True
    nodes = curve_mat.node_tree.nodes
    nodes.clear()
    
    # 创建纯白自发光材质
    emission = nodes.new('ShaderNodeEmission')
    emission.inputs['Color'].default_value = (1, 1, 1, 1)  # 纯白色
    emission.inputs['Strength'].default_value = 5  # 增加发光强度
    
    output = nodes.new('ShaderNodeOutputMaterial')
    curve_mat.node_tree.links.new(emission.outputs['Emission'], output.inputs['Surface'])
    
    return curve_mat

def render_mask(scene, camera_obj, output_path):
    
    # 保存原始设置
    original_camera = scene.camera
    original_engine = scene.render.engine
    original_film_transparent = scene.render.film_transparent
    
    # 设置当前相机
    scene.camera = camera_obj
    
    # 使用Cycles渲染器
    scene.render.engine = 'CYCLES'  
    bpy.context.scene.cycles.device = 'GPU1'
    # 关闭Cycles的抗锯齿
    # 降低采样数到最小值
    bpy.context.scene.cycles.samples = 1
    
    # 关闭抗锯齿过滤
    bpy.context.scene.cycles.filter_width = 0.0

    scene.cycles.max_bounces = 0
    scene.cycles.diffuse_bounces = 0
    scene.cycles.glossy_bounces = 0
    scene.cycles.transparent_max_bounces = 0
    scene.cycles.transmission_bounces = 0
    scene.cycles.volume_bounces = 0
    scene.cycles.caustics_reflective = False
    scene.cycles.caustics_refractive = False

    scene.render.film_transparent = True  # 透明背景

    # 设置曲线材质
    curve_mat = setup_mask_materials()
    
    # 应用材质到曲线
    for obj in scene.objects:
        if obj.type == 'CURVE':
            if obj.data.materials:
                obj.data.materials[0] = curve_mat
            else:
                obj.data.materials.append(curve_mat)
    
    # 设置渲染参数
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'BW'  # 黑白模式
    scene.render.image_settings.color_depth = '8'
    
    # 禁用所有灯光
    for light in bpy.data.lights:
        light.energy = 0
    
    # 设置世界背景为完全透明
    if scene.world is None:
        scene.world = bpy.data.worlds.new("Transparent_World")
    scene.world.use_nodes = True
    nodes = scene.world.node_tree.nodes
    nodes.clear()
    bg_node = nodes.new('ShaderNodeBackground')
    bg_node.inputs['Color'].default_value = (0, 0, 0, 1)  # 黑色
    bg_node.inputs['Strength'].default_value = 0.0  # 无光照
    output = nodes.new('ShaderNodeOutputWorld')
    scene.world.node_tree.links.new(bg_node.outputs['Background'], output.inputs['Surface'])
    
    # 使用合成节点确保输出alpha通道
    scene.use_nodes = True
    node_tree = scene.node_tree
    nodes = node_tree.nodes
    links = node_tree.links
    
    # 清除现有节点
    nodes.clear()
    
    # 添加渲染层和输出节点
    render_layers = nodes.new('CompositorNodeRLayers')
    composite = nodes.new('CompositorNodeComposite')
    
    # 连接节点
    links.new(render_layers.outputs['Image'], composite.inputs['Image'])
    
    # 渲染
    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    
    # 恢复原始设置
    scene.camera = original_camera
    scene.render.engine = original_engine
    scene.render.film_transparent = original_film_transparent
    
def add_curve_thickness_bevel(curve_obj, thickness, resolution=12):
    """
    使用倒角为曲线添加粗度（最简单的方法）
    
    参数:
    - curve_obj: 曲线对象
    - thickness: 曲线直径
    - resolution: 倒角分辨率（越高越平滑）
    """
    if curve_obj.type != 'CURVE':
        print("错误：对象不是曲线类型")
        return curve_obj
    
    curve_obj.data.bevel_depth = thickness / 2  # bevel_depth 是半径，所以要除以2
    curve_obj.data.bevel_resolution = resolution
    curve_obj.data.fill_mode = 'FULL'  # 填充模式
    
    print(f"已为曲线添加倒角厚度: {thickness}, 分辨率: {resolution}")
    return curve_obj

def setup_black_polymer_guidewire_material():
    """
    设置黑色高分子涂层的导丝材质
    """
    # 创建高分子涂层材质
    polymer_mat = bpy.data.materials.new(name="Black_Polymer_Guidewire")
    polymer_mat.use_nodes = True
    nodes = polymer_mat.node_tree.nodes
    nodes.clear()
    
    # 创建材质节点
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    
    # 黑色高分子涂层参数 - 只使用最基本的参数
    bsdf.inputs['Base Color'].default_value = (0.05, 0.05, 0.06, 1.0)  # 深黑色
    bsdf.inputs['Metallic'].default_value = 1  # 金属度
    bsdf.inputs['Roughness'].default_value = 0.3  # 中等粗糙度
    
    output = nodes.new('ShaderNodeOutputMaterial')
    links = polymer_mat.node_tree.links
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    # 调整节点布局
    bsdf.location = (300, 0)
    output.location = (500, 0)
    
    return polymer_mat


def apply_material_to_guidewire(curve_obj, material):
    """
    将材质应用到导丝
    """
    if curve_obj.data.materials:
        curve_obj.data.materials[0] = material
    else:
        curve_obj.data.materials.append(material)
    
    print("已应用黑色高分子涂层材质到导丝")

def setup_medical_lighting():
    """
    设置医疗环境的光照（更清晰展示导丝细节）
    """
    # 删除现有灯光
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        if obj.type == 'LIGHT':
            obj.select_set(True)
    bpy.ops.object.delete()
    
    # 设置医疗风格的HDRI环境
    if bpy.context.scene.world is None:
        bpy.context.scene.world = bpy.data.worlds.new("Medical_World")
    
    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    
    
    background = nodes.new('ShaderNodeBackground')
    background.inputs['Color'].default_value = (1.0, 0.95, 0.85, 1.0)  # 白黄色
    background.inputs['Strength'].default_value = 3.0  # 更强的环境光
    output = nodes.new('ShaderNodeOutputWorld')
    links.new(background.outputs['Background'], output.inputs['Surface'])
    
    # 创建专业的医疗照明
    def create_medical_light(name, location, rotation, energy, size=0.1):
        light_data = bpy.data.lights.new(name=name, type='AREA')
        light_obj = bpy.data.objects.new(name, light_data)
        light_obj.location = location
        light_obj.rotation_euler = rotation
        light_data.energy = energy
        light_data.size = size
        light_data.shape = 'RECTANGLE'
        light_data.color = (1, 1, 0.95)  # 略带暖白色
        bpy.context.collection.objects.link(light_obj)
        return light_obj
    
    # 主照明 - 从上方照射
    create_medical_light("Main_Light", (0, -2, 8), (1.2, 0, 0), 400, 4)
    
    # 侧光 - 突出轮廓
    create_medical_light("Side_Light", (6, 0, 5), (1.57, 0, 0.8), 800, 3)
    
    # 补光 - 减少阴影
    create_medical_light("Fill_Light", (-4, 4, 3), (1.0, 0, -0.5), 400, 2)
    
    return world

def setup_rendering_settings():
    """
    设置渲染参数
    """
    # 使用Cycles渲染器
    scene.render.engine = 'CYCLES'  
    bpy.context.scene.cycles.device = 'GPU1'

    scene = bpy.context.scene
    scene.cycles.samples = 128  # 采样数
    scene.cycles.max_bounces = 8  # 最大反弹次数
    
    # 渲染设置

    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.color_depth = '16'
    scene.render.film_transparent = False
    
    return scene

def render_realistic_view(scene, camera_obj, output_path):
    """
    渲染真实的导丝图像（彩色RGB图像）
    """
    # 保存原始设置
    original_camera = scene.camera
    original_engine = scene.render.engine
    
    # 设置当前相机
    scene.camera = camera_obj
      
    # 设置输出路径和格式
    scene.render.filepath = output_path
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.color_depth = '16'
    
    # 确保输出目录存在
    output_dir = os.path.dirname(bpy.path.abspath(output_path))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 渲染
    print(f"开始渲染真实图像到: {output_path}")
    bpy.ops.render.render(write_still=True)
    print("真实图像渲染完成")
    
    # 恢复原始设置
    scene.camera = original_camera
    scene.render.engine = original_engine
    
    return output_path
def sample_curve_surface_and_save_ply(curve_obj, num_points=10000, output_path="//curve_surface_points.ply"):
    """
    对曲线表面进行随机均匀采样并保存为PLY点云
    """

    # 创建临时网格对象来获取表面几何
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated_obj = curve_obj.evaluated_get(depsgraph)
    mesh = bpy.data.meshes.new_from_object(evaluated_obj)
    
    # 创建BMesh来操作网格
    bm = bmesh.new()
    bm.from_mesh(mesh)
    
    # 确保面是三角化的，便于面积计算
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    
    # 关键修复：将面转换为列表，避免索引问题（替代ensure_lookup_table）
    faces_list = list(bm.faces)
    
    # 计算每个面的面积
    face_areas = []
    total_area = 0.0
    
    for face in faces_list:
        area = face.calc_area()
        face_areas.append(area)
        total_area += area
    
    print(f"总表面积: {total_area:.6f}")
    print(f"面数: {len(faces_list)}")
    
    # 基于面积比例进行采样
    sampled_points = []
    sampled_normals = []
    
    for _ in range(num_points):
        # 根据面积比例随机选择面
        random_value = random.uniform(0, total_area)
        cumulative_area = 0.0
        selected_face = None
        
        for i, area in enumerate(face_areas):
            cumulative_area += area
            if cumulative_area >= random_value:
                # 从列表中获取面，避免直接用索引访问bm.faces
                selected_face = faces_list[i]
                break
        
        if selected_face is None:
            selected_face = faces_list[-1]
        
        # 在选中的面上随机采样一个点
        # 修复：将顶点转换为列表，避免索引问题
        verts = list(selected_face.verts)
        if len(verts) == 3:  # 三角形
            u = random.random()
            v = random.random()
            if u + v > 1:
                u = 1 - u
                v = 1 - v
            w = 1 - u - v
            
            point = (verts[0].co * u + 
                    verts[1].co * v + 
                    verts[2].co * w)
        else:  # 其他多边形
            point = selected_face.calc_center_median()
        
        # 获取法线
        normal = selected_face.normal
        
        sampled_points.append(point)
        sampled_normals.append(normal)
    
    # 转换为世界坐标
    world_matrix = curve_obj.matrix_world
    world_points = [world_matrix @ point for point in sampled_points]
    world_normals = [world_matrix.to_3x3() @ normal for normal in sampled_normals]
    
    # 保存为PLY文件
    save_points_as_ply(world_points, filepath=output_path, normals=world_normals)
    
    # 清理
    bm.free()
    bpy.data.meshes.remove(mesh)
    
    print(f"成功采样 {num_points} 个点，保存到: {output_path}")
    return world_points, world_normals

def save_points_as_ply(points, filepath, normals=None):
    """
    将点和可选的的法线保存为PLY格式
    
    参数:
    points: 点坐标列表 (可以是numpy数组或Blender Vector对象)
    normals: 法线列表 (可以是numpy数组或Blender Vector对象)，可选
    filepath: 保存的文件路径
    """
    import struct
    import os
    
    # 确保路径是绝对路径
    abs_path = bpy.path.abspath(filepath)
    
    # 确保目录存在
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    
    with open(abs_path, 'wb') as f:
        # PLY文件头
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {len(points)}\n".encode())
        f.write(b"property float x\n")
        f.write(b"property float y\n")
        f.write(b"property float z\n")
        
        # 如果有法线信息，添加法线属性
        has_normals = normals is not None and len(normals) == len(points)
        if has_normals:
            f.write(b"property float nx\n")
            f.write(b"property float ny\n")
            f.write(b"property float nz\n")
        
        f.write(b"end_header\n")
        
        # 写入点数据
        for i in range(len(points)):
            point = points[i]
            
            # 处理不同类型的点数据
            if hasattr(point, 'x'):  # Blender Vector 对象
                x, y, z = point.x, point.y, point.z
            else:  # NumPy 数组或其他可索引对象
                x, y, z = point[0], point[1], point[2]
            
            # 写入坐标
            f.write(struct.pack('<fff', x, y, z))
            
            # 如果有法线信息，写入法线
            if has_normals:
                normal = normals[i]
                
                # 处理不同类型的法线数据
                if hasattr(normal, 'x'):  # Blender Vector 对象
                    nx, ny, nz = normal.x, normal.y, normal.z
                else:  # NumPy 数组或其他可索引对象
                    nx, ny, nz = normal[0], normal[1], normal[2]
                
                # 归一化法线
                length = math.sqrt(nx*nx + ny*ny + nz*nz)
                if length > 0:
                    nx, ny, nz = nx/length, ny/length, nz/length
                
                f.write(struct.pack('<fff', nx, ny, nz))
    
    print(f"PLY文件已保存: {abs_path}")
    if has_normals:
        print(f"包含 {len(points)} 个点和法线")
    else:
        print(f"包含 {len(points)} 个点（无法线信息）")
def sample_bezier_curve_uniform(curve_obj, num_samples=100):
    """
    在贝塞尔曲线上均匀等间距采样3D点
    
    参数:
    curve_obj: Blender中的曲线对象
    num_samples: 采样点数量
    
    返回:
    numpy数组，形状为(num_samples, 3)，包含曲线上的3D坐标
    """
    # 确保是曲线对象
    if curve_obj.type != 'CURVE':
        raise ValueError("提供的对象不是曲线")
    
    # 获取曲线数据
    curve = curve_obj.data
    spline = curve.splines[0]  # 假设只有一条样条线
    
    # 将曲线转换为多边形以便均匀采样
    # 创建一个临时副本进行评估
    depsgraph = bpy.context.evaluated_depsgraph_get()
    curve_eval = curve_obj.evaluated_get(depsgraph)
    mesh = curve_eval.to_mesh()
    
    # 获取所有顶点
    vertices = [v.co for v in mesh.vertices]
    
    # 计算总长度和每个线段的长度
    total_length = 0
    segment_lengths = []
    for i in range(len(vertices) - 1):
        length = (vertices[i+1] - vertices[i]).length
        segment_lengths.append(length)
        total_length += length
    
    # 均匀采样
    sampled_points = []
    sample_distance = total_length / (num_samples - 1)
    current_distance = 0
    current_segment = 0
    segment_start = 0
    segment_end = segment_lengths[0]
    
    for i in range(num_samples):
        # 找到当前采样点所在的线段
        while current_distance > segment_end and current_segment < len(segment_lengths) - 1:
            current_segment += 1
            segment_start = segment_end
            segment_end += segment_lengths[current_segment]
        
        # 在线段内插值
        segment_progress = (current_distance - segment_start) / segment_lengths[current_segment]
        point = vertices[current_segment] * (1 - segment_progress) + vertices[current_segment + 1] * segment_progress
        
        # 转换到世界坐标
        world_point = curve_obj.matrix_world @ point
        sampled_points.append([world_point.x, world_point.y, world_point.z])
        
        current_distance += sample_distance
    
    # 清理临时网格
    curve_eval.to_mesh_clear()
    
    return np.array(sampled_points)    
def generate_single_sample(sample_dir):
    """生成单个样本并保存到指定目录"""
    # 删除现有对象
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # 创建曲线
    curve_data = create_guidewire()
    curve_obj = curve_data['curve_object']

    # 采样曲线并保存点云
    sampled_points = sample_bezier_curve_uniform(curve_obj, num_samples=2000)
    ply_path = os.path.join(sample_dir, "curve_points.ply")
    save_points_as_ply(sampled_points, ply_path)

    # 设置相机
    cameras = setup_cameras(curve_obj)
    
    # 设置渲染参数
    scene = bpy.context.scene
    scene.render.resolution_x = 512
    scene.render.resolution_y = 2048
    
    add_curve_thickness_bevel(curve_obj, thickness=0.1, resolution=12)

    # 拍摄相机1的照片
    mask1_path = os.path.join(sample_dir, "mask_cam1")
    render_mask(scene, cameras['cam1']['object'], mask1_path)
    
    # 拍摄相机2的照片
    mask2_path = os.path.join(sample_dir, "mask_cam2")
    render_mask(scene, cameras['cam2']['object'], mask2_path)

    
    # 保存相机参数到文件
    camera_data = {
        'cam1': {
            'intrinsic': [list(row) for row in cameras['cam1']['params']['intrinsic']],
            'extrinsic': [list(row) for row in cameras['cam1']['params']['extrinsic']],
            'resolution': cameras['cam1']['params']['resolution']
        },
        'cam2': {
            'intrinsic': [list(row) for row in cameras['cam2']['params']['intrinsic']],
            'extrinsic': [list(row) for row in cameras['cam2']['params']['extrinsic']],
            'resolution': cameras['cam2']['params']['resolution']
        }
    }
    
    camera_json_path = os.path.join(sample_dir, "camera_params.json")
    with open(camera_json_path, 'w') as f:
        json.dump(camera_data, f, indent=4)
    
    print(f"相机参数已保存到 {camera_json_path}")
    
    add_curve_thickness_bevel(curve_obj, thickness=0.4, resolution=12)
    polymer_mat = setup_black_polymer_guidewire_material()
    apply_material_to_guidewire(curve_obj, polymer_mat)
    setup_medical_lighting()
    
    # 设置渲染参数
    scene = setup_rendering_settings()
    # 渲染真实图像
    img1_path = os.path.join(sample_dir, "polymer_guidewire_cam1.png")
    render_realistic_view(scene, cameras['cam1']['object'], img1_path)
    
    img2_path = os.path.join(sample_dir, "polymer_guidewire_cam2.png")
    render_realistic_view(scene, cameras['cam2']['object'], img2_path)
    
    return {
        'curve': curve_data,
        'cameras': cameras
    }

def generate_batch_dataset(num_samples, base_dir="dataset"):
    """
    批量生成数据集
    
    参数:
        num_samples: 要生成的样本数量
        base_dir: 主数据集文件夹名称，默认为"dataset"
    """
    # 创建主数据集文件夹
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    
    # 生成指定数量的样本
    for i in range(num_samples):
        # 创建样本子文件夹（格式sample_0001）
        sample_folder = os.path.join(base_dir, f"sample_{i+1:04d}")
        Path(sample_folder).mkdir(parents=True, exist_ok=True)
        
        print(f"正在生成样本 {i+1}/{num_samples}，保存至: {sample_folder}")
        
        # 生成并保存单个样本
        generate_single_sample(sample_folder)


def main():    # 设置随机种子以确保可重复性
    random.seed(42)
    np.random.seed(42)
    generate_batch_dataset(num_samples=2, base_dir="C:/Users/Dell/Desktop/dataset guidewire/dataset")
    return "完成"
# 运行主函数
if __name__ == "__main__":
    result = main()