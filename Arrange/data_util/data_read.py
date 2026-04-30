
from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import copy
import torch.nn.utils.rnn as rnn_utils
#import graphto3d.dataset.util as util
from tqdm import tqdm
import json
#from graphto3d.helpers.psutil import FreeMemLinux
#from graphto3d.helpers.util import normalize_box_params
import random
import pickle

class my_Dataset(data.Dataset):
    def __init__(self,root,with_feats,if_debug):
        self.data_root=root
        self.with_feats=with_feats
        self.debug=if_debug
        self.goal_files=[]
        
    def get_goal_files(self):
        goal_files=[]
        '''检索含目标场景信息的文件，返回含目标文件名的列表'''
        for filename in os.listdir(self.data_root):
            if '_goal_view-2.json' in filename:
                goal_files.append(filename)
                self.goal_files.append(filename)
        return goal_files
    
    def get_label_name_to_global_id(self,file_path):
        labelName2InstanceId = {} #{'teapot_2': 8}
        instanceId2class = {} #{ 1: 'fork'}

        with open(file_path, 'r') as f:
            dict_out = json.load(f)
            for obj in dict_out['objects']:
                labelName2InstanceId[obj['name']] = obj['global_id']
                instanceId2class[obj['global_id']] = obj['class']
        if not labelName2InstanceId or not instanceId2class:
            raise ValueError(f"labelName2classId or classId2ClassName of {file_path} shouldn't be empty!")
        return labelName2InstanceId, instanceId2class
    

    def instance_id2mask(self,path_idx):

        #!labelName2InstanceId, e.g. {'plate_6': 10, '2362ec480b3e9baa4fd5721982c508ad_support_table': 109, 'fork_1': 1, 'knife_1': 2}
        #!instanceId2class {10: 'plate', 109: 'support_table', 1: 'fork', 2: 'knife'}
        labelName2InstanceId, instanceId2class = self.get_label_name_to_global_id(path_idx)
        keys = list(instanceId2class.keys()) #! all instance_ids  目标场景所有物体的全域id

        instance2mask = {}
        instance2mask[0] = 0 
        counter=0
        #! instance2mask {0: 0, 1: 1, 2: 2, 109: 3, 10: 4}
        for key in keys: #! 所有的instance_ids e.g., [1,2,109,10]
            # get objects from the selected list of classes of 3dssg
            scene_instance_id = key #!1
            instance2mask[scene_instance_id] = counter + 1 #
            counter += 1
        return instance2mask
           

    def __getitem__(self, idx):
        class_ids=[]#存储当前场景下物体类别的id
        bbox_es=[]#存储当前场景下物体的包围盒数据
        locations=[]
        angles=[]
        triples=[]


        path_idx=os.path.join(self.data_root,self.goal_files[idx])#布局信息文件路径
        scene_graph_path = path_idx.replace('_goal_view-2', '_goal_scene_graph')#图信息文件路径
        instance2mask=self.instance_id2mask(path_idx)
       
        with open(path_idx, 'r', encoding='utf-8') as file:
            data = json.load(file)#
            object_data=data["objects"]
            for f in object_data:#每个物体的信息
                class_ids.append(f['class_id'])
                bbox_es.append(f['param6'])
                locations.append(f['location'])
                angles.append(f['quaternion_xyzw'])
        with open(scene_graph_path, 'r', encoding='utf-8') as file: 
            data_graph=json.load(file)
            for r in data_graph["relationships"]:
                if r[0] in instance2mask.keys() and r[1] in instance2mask.keys():  #r[0], r[1] -> instance_id
                    subject = instance2mask[r[0]]-1 #! instance2mask[r[0] -> 实例在场景中的编号/索引 - 1, 最后一个node '_scene_' 放最后
                    object = instance2mask[r[1]]-1
                    predicate = r[2] +1
                    if subject >= 0 and object >= 0:
                         triples.append([subject, predicate, object])
                else:
                    continue     
        datum = {
            "class_ids": class_ids,
            "bbox_es": bbox_es,
            "locations": locations,
            'quaternion_xyzw':angles,
            "triples":triples
        }
        return self.convert_tensor(datum)
        #return datum  

            

        
    
    def __len__(self):
        return len(self.goal_files)
    

    def convert_tensor(self,datum):
        
        tensors={
            # "class_ids": torch.LongTensor(np.array(datum["class_ids"])),
            # "bbox_es": torch.FloatTensor(np.array(datum["bbox_es"])),
            # "locations": torch.FloatTensor(np.array(datum["locations"])),
            # "quaternion_xyzw":torch.FloatTensor(np.array(datum['quaternion_xyzw'])),
            # "triples":torch.FloatTensor(np.array(datum['triples']))

            "class_ids": torch.from_numpy(np.array(datum["class_ids"],dtype=np.int64)),
            "bbox_es": torch.from_numpy(np.array(datum["bbox_es"],dtype=np.float32)),
            "locations": torch.from_numpy(np.array(datum["locations"],dtype=np.float32)),
            "quaternion_xyzw":torch.from_numpy(np.array(datum['quaternion_xyzw'],dtype=np.float32)),
            "triples":torch.from_numpy(np.array(datum['triples'],dtype=np.int64))
         }
        return tensors
    
    def quaternion_to_sin_cos(self,qw):
        """
    将四元数转换为正弦和余弦值。
    参数:
    q -- 四元数，格式为 (w, x, y, z)
    返回:
    sin_theta -- 正弦值
    cos_theta -- 余弦值
         """
        w=qw[0]
    # 计算旋转角度
        theta = torch.tensor(2 * np.arccos(w))  # 计算旋转角度 (弧度)
    # 计算正弦和余弦值
        sin_theta = torch.sin(theta / 2)
        cos_theta = torch.cos(theta / 2)
        angles = torch.stack([sin_theta, cos_theta]).unsqueeze(0) 
        return angles

    def collate_fn(self,batch):
    # 这里的 batch 是从 DataLoader 中获取的一批样本
        
        # "class_ids": rnn_utils.pad_sequence([item['class_ids'] for item in batch], batch_first=True),
        # "bbox_es": rnn_utils.pad_sequence([item['bbox_es'] for item in batch], batch_first=True),
        # "locations": rnn_utils.pad_sequence([item['locations'] for item in batch], batch_first=True),
        # "quaternion_xyzw": rnn_utils.pad_sequence([item['quaternion_xyzw'] for item in batch], batch_first=True),
        # "triples":rnn_utils.pad_sequence([item['triples'] for item in batch],batch_first=True)
        
        all_objs, all_boxes, all_triples = [], [], []
        all_locations,all_angles=[],[]
        for i in range(len(batch)):
            (objs, triples, boxes) = batch[i]['class_ids'], batch[i]['triples'], batch[i]['bbox_es']
            locations,quaternion_xyzw= batch[i]["locations"],batch[i]["quaternion_xyzw"]
            #angles=self.quaternion_to_sin_cos(quaternion_xyzw)
            all_triples.append(triples)
            all_objs.append(objs)
            all_boxes.append(boxes)
            all_angles.append(quaternion_xyzw)
            all_locations.append(locations)
        all_objs = torch.cat(all_objs)
        all_boxes=torch.cat(all_boxes)
        all_triples=torch.cat(all_triples)
        all_locations=torch.cat(all_locations)
        # 将列表转换为张量
       
        all_angles=torch.cat(all_angles)

        return {"class_ids":all_objs, 
                "triples":all_triples,
                "bbox_es":all_boxes,
                "locations":all_locations,
                "quaternion_xyzw":all_angles
                }
        
        
        
        
        
        
        
if __name__=="__main__":
    root="/remote-home/2332082/data/sgbot_dataset/raw/7b4acb843fd4b0b335836c728d324152"
    root2="D:\\pythonhomework\\Project_Repetition\\SG-Bot\\sgbot_dataset\\7b4acb843fd4b0b335836c728d324152"
    #print(os.listdir(root))
    my_data=my_Dataset(root,None,None)
    files=my_data.get_goal_files()
    # for i in files:
    #     print(i)
    # print(type(files))

    batch_size = 4  # 设置批次大小
    data_loader = data.DataLoader(my_data, batch_size=batch_size, shuffle=True,collate_fn=my_data.collate_fn)

    # 输出一个批次的数据
    for batch in data_loader:
        print(batch)
        print(batch['bbox_es'].shape)
        print(batch['angles'].shape)
        break  # 只输出一个批次的数据
