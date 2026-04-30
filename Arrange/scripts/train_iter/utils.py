import yaml
from yaml import Loader
import cv2
import numpy as np
import os
import torch
import trimesh
import seaborn as sns
import json 
import pyrender
from PIL import Image
def normalize_box_params(box_params, scale=3):
    """ Normalize the box parameters for more stable learning utilizing the accumulated dataset statistics

    :param box_params: float array of shape [7] containing the box parameters
    :param scale: float scalar that scales the parameter distribution
    :return: normalized box parameters array of shape [7]
    """
    mean = np.array([ 0.2610482 ,  0.22473196,  0.14623462,  0.0010283 , -0.02288815 , 0.20876316])#np.array([ 2.42144732e-01,  2.35105852e-01,  1.53590141e-01, -1.54968627e-04, -2.68763962e-02,  2.23784580e-01 ])
    std = np.array([0.31285113, 0.21937416, 0.17070778, 0.14874465, 0.1200992,  0.11501499])#np.array([ 0.27346058, 0.23751527, 0.18529049, 0.12504842, 0.13313938 ,0.12407406 ])

    return scale * ((box_params - mean) / std)

def batch_torch_denormalize_box_params(box_params, scale=3):
    """ Denormalize the box parameters utilizing the accumulated dateaset statistics

    :param box_params: float tensor of shape [N, 6] containing the 6 box parameters, where N is the number of boxes
    :param scale: float scalar that scales the parameter distribution
    :return: float tensor of shape [N, 6], the denormalized box parameters
    """

    mean = torch.tensor([ 0.2610482 ,  0.22473196,  0.14623462,  0.0010283 , -0.02288815 , 0.20876316]).reshape(1,-1).float().cuda()#torch.tensor([ 2.42144732e-01,  2.35105852e-01,  1.53590141e-01, -1.54968627e-04, -2.68763962e-02,  2.23784580e-01 ]).reshape(1,-1).float().cuda()
    std = torch.tensor([0.31285113, 0.21937416, 0.17070778, 0.14874465, 0.1200992,  0.11501499]).reshape(1,-1).float().cuda()#torch.tensor([ 0.27346058, 0.23751527, 0.18529049, 0.12504842, 0.13313938 ,0.12407406 ]).reshape(1,-1).float().cuda()

    return (box_params * std) / scale + mean

def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=Loader)
    return config

def render(name_dict,obj_ids, predBoxes, predAngles, render_type='scene',
               store_img=False, render_boxes=False, demo=False, visual=False, without_lamp=False,
               str_append="", mani=0, missing_nodes=None, manipulated_nodes=None, objs_before=None, store_path=None):
    os.makedirs(store_path,exist_ok=True)
    if render_type not in ['txt2shape', 'scene', 'onlybox']:
        raise ValueError('Render type needs to be either set to txt2shape or retrieval or onlybox.')
    color_palette = np.array(sns.color_palette('hls', 100))


    box_and_angle = torch.cat([predBoxes.float(), predAngles.float()], dim=-1)

    obj_n = len(box_and_angle)

    # mesh_dir = os.path.join(store_path, render_type, 'object_meshes', "hello")
    # os.makedirs(mesh_dir, exist_ok=True)
    
    

    if render_type == 'onlybox':
        lamp_mesh_list, trimesh_meshes = get_bbox(box_and_angle,obj_ids, colors=color_palette[obj_ids])
    
    elif render_type=="scene":
        trimesh_meshes, raw_meshes = get_database_objects(name_dict,box_and_angle, 
                                                                         render_boxes=render_boxes,
                                                                            colors=color_palette 
                                                                            )
        

    else:
        raise NotImplementedError



    scene = trimesh.Scene(trimesh_meshes)
    scene_path = os.path.join(store_path, render_type)
    if len(str_append) > 0:
        render_type += str_append
    os.makedirs(scene_path, exist_ok=True)
    scene.export(os.path.join(scene_path, "{0}_{1}.glb".format('hello_new', render_type)))

    
    # scene.show()


    img_path = os.path.join(store_path, render_type, "render_imgs")
    os.makedirs(img_path, exist_ok=True)
    color_img = render_img(trimesh_meshes)
    color_bgr = cv2.cvtColor(color_img, cv2.COLOR_RGBA2BGR)
    file_name = "hello"
    if len(str_append) > 0:
            file_name += str_append
    cv2.imwrite(os.path.join('/remote-home/2332082/ArrangeBot/Arrange/scripts/train_iter', f'{file_name}.png'), color_bgr)
    print("已生成图片")
    if mani==1:
        return trimesh_meshes


def get_database_objects(name_dict,boxes, render_boxes=False, colors=None):
    
    # os.makedirs(mesh_dir, exist_ok=True)
   
    colors = iter(colors)
    
    trimesh_meshes = []
    raw_meshes = []
    print("boxes_num",boxes.shape[0])
    print("dict_len:",len(name_dict.keys()))

    model_base_path = "/remote-home/2332082/data/sgbot_dataset/models"
    assert boxes.shape[0]==len(name_dict)
    keys=list(name_dict.keys())
    for i in range(len(keys)):

        name=name_dict[keys[i]]
        if "support_table" not in name:
            model_path = os.path.join(model_base_path,keys[i],f"{name}.obj")
        
            texture_path = os.path.join(model_base_path,keys[i], f"{name}.png")
        else:
            name=name.replace("_support_table","")
            model_path = os.path.join(model_base_path,keys[i],f"{name}.obj")
            texture_path = os.path.join(model_base_path,keys[i],"aefc493c1bfb5fa2.png")



        color = next(colors)

        # Load the furniture and scale it as it is given in the dataset
        tr_mesh = trimesh.load(model_path, force="mesh")
        tr_mesh = trimesh.Trimesh(vertices=tr_mesh.vertices, faces=tr_mesh.faces, process=False)
        
      
        texture_image = Image.open(texture_path)
        texture_image = np.array(texture_image)  # Convert to numpy array


        texture_visuals = trimesh.visual.TextureVisuals(image=texture_image)

        # Apply texture visuals to the mesh
        tr_mesh.visual = texture_visuals

        tr_mesh.visual.vertex_colors = color
        tr_mesh.visual.face_colors = color

        raw_meshes.append(tr_mesh.copy())

        #tr_mesh.export(os.path.join(mesh_dir, query_label+'_'+str(cat_ids[j])+'_'+str(instance_id)+".obj"))
   
    # tr_mesh.visual.material.image = Image.open(texture_path)
        theta = boxes[i, -1].item() * (np.pi / 180)
        R = np.zeros((3, 3))
        R[0, 0] = np.cos(theta)
        R[0, 2] = -np.sin(theta)
        R[2, 0] = np.sin(theta)
        R[2, 2] = np.cos(theta)
        R[1, 1] = 1.
        t = boxes[i, 3:6].detach().cpu().numpy()
        tr_mesh.vertices[...] = tr_mesh.vertices.dot(R) + t#计算旋转矩阵和位移，将物体的位置和方向应用到模型顶点
        trimesh_meshes.append(tr_mesh)
        if render_boxes:
            box_points = params_to_8points_3dfront(boxes[i], degrees=True)
            trimesh_meshes.append(create_bbox_marker(box_points, tube_radius=0.006, color=color))


    return trimesh_meshes, raw_meshes

def get_bbox(boxes,obj_ids, colors):
    trimesh_meshes = []
    colors = iter(colors)
    lamp_mesh_list=[]
    print(boxes)
    print(boxes.shape)
    classes=['_scene_',
    'bowl',
    'box',
    'can',
    'cup',
    'fork',
    'knife',
    'pitcher',
    'plate',
    'support_table',
    'tablespoon',
    'teapot',
    'teaspoon',
    'obstacle']
    for j in range(0, boxes.shape[0]):
        query_label = classes[obj_ids[j]].strip('\n')
        print("rendering", classes[obj_ids[j]].strip('\n') )
        if query_label == '_scene_' or query_label == 'obstacle':
            continue
        box_points = params_to_8points_3dfront(boxes[j], degrees=True)
        print('box_points',box_points)
        trimesh_meshes.append(create_bbox_marker(box_points, tube_radius=0.02, color=next(colors)))
        # if query_label == 'nightstand':
        #     trimesh_meshes.pop()
        
    

    return lamp_mesh_list, trimesh_meshes

def create_bbox_marker(corner_points, color=[0, 0, 255], tube_radius=0.002, sections=4):
    """Create a 3D mesh visualizing a bbox. It consists of 12 cylinders.

    Args:
        corner_points
        color (list, optional): RGB values of marker. Defaults to [0, 0, 255].
        tube_radius (float, optional): Radius of cylinders. Defaults to 0.001.
        sections (int, optional): Number of sections of each cylinder. Defaults to 4.

    Returns:
        trimesh.Trimesh: A mesh.
    """
    edges = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]
    bbox_edge_list = []
    for edge in edges:
        bbox_edge = trimesh.creation.cylinder(radius=tube_radius,sections=sections,segment=[corner_points[edge[0]],corner_points[edge[1]]])
        bbox_edge_list.append(bbox_edge)

    tmp = trimesh.util.concatenate(bbox_edge_list)
    tmp.visual.face_colors = color

    # z axis to x axis
    # R = np.array([[0,0,1],[1,0,0],[0,1,0]]).reshape(3,3)
    # t =  np.array([0, 0, -1.12169998e-01]).reshape(3,1)
    #
    # T = np.r_[np.c_[np.eye(3), t], [[0, 0, 0, 1]]]
    # tmp.apply_transform(T)

    return tmp


def params_to_8points_3dfront(box, degrees=False):
    """ Given bounding box as 7 parameters: l, h, w, cx, cy, cz, z, compute the 8 corners of the box
    """
    l, h, w, px, py, pz, angle = box
    points = []
    for i in [-1, 1]:
        for j in [0, 1]:
            for k in [-1, 1]:
                points.append([l.item()/2 * i, h.item() * j, w.item()/2 * k])
    points = np.asarray(points)
    points = points.dot(get_rotation_3dfront(angle.item(), degree=degrees))
    points += np.expand_dims(np.array([px.item(), py.item(), pz.item()]), 0)
    return points

def get_rotation_3dfront(y, degree=True):
    if degree:
        y = np.deg2rad(y)
    rot = np.array([[np.cos(y),     0,  -np.sin(y)],
                    [       0 ,     1,           0],
                    [np.sin(y),     0,   np.cos(y)]])
    return rot



def render_img(trimesh_meshes):
    scene = pyrender.Scene()
    renderer = pyrender.OffscreenRenderer(viewport_width=256, viewport_height=256)
    for tri_mesh in trimesh_meshes:
        pyrender_mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=False)
        scene.add(pyrender_mesh)

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 2)

    # set up positions and the origin
    camera_location = np.array([0.0, 8.0, 0.0])  # y axis
    look_at_point = np.array([0.0, 0.0, 0.0])
    up_vector = np.array([0.0, 0.0, -1.0])  # -z axis

    camera_direction = (look_at_point - camera_location) / np.linalg.norm(look_at_point - camera_location)
    right_vector = np.cross(camera_direction, up_vector)
    up_vector = np.cross(right_vector, camera_direction)

    camera_pose = np.identity(4)
    camera_pose[:3, 0] = right_vector
    camera_pose[:3, 1] = up_vector
    camera_pose[:3, 2] = -camera_direction
    camera_pose[:3, 3] = camera_location
    scene.add(camera, pose=camera_pose)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    scene.add(light, pose=camera_pose)

    point_light = pyrender.PointLight(color=np.ones(3), intensity=20.0)
    scene.add(point_light, pose=camera_pose)
    color, depth = renderer.render(scene)
    return color