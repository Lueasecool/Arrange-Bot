from __future__ import print_function
import sys
from tqdm import tqdm
sys.path.append('./Arrange')
sys.path.append('./Arrange/scripts')
import open3d as o3d # open3d needs to be imported before other packages!
import argparse
import os
import random
import numpy as np
import torch
import torch.nn.parallel
import json
import torch.utils.data
from omegaconf import OmegaConf
from data_util.data_readfull import my_Dataset
from train_iter.utils import batch_torch_denormalize_box_params
from utils import render


from scripts.diffusion_Unet.model_unet import DiffusionScene
from scripts.train_iter.utils import load_config
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="/remote-home/2332082/data/sgbot_dataset/raw", help="dataset path")
parser.add_argument('--filepath', type=str, default="/remote-home/2332082/data/sgbot_dataset/train_scenes.txt", help="dataset path")
parser.add_argument('--with_feats', type=bool, default=False, help="Load Feats directly instead of points.")
parser.add_argument('--if_debug',type=bool, default=False)

parser.add_argument('--manipulate', default=True, type=bool)
parser.add_argument('--exp', default='../experiments/layout_test', help='experiment name')
parser.add_argument('--epoch', type=str, default='100', help='saved epoch')
parser.add_argument('--render_type', type=str, default='txt2shape', help='retrieval, txt2shape, onlybox, echoscene')
parser.add_argument('--gen_shape', default=False, type=bool, help='infer diffusion')
parser.add_argument('--visualize', default=False, type=bool)
parser.add_argument('--export_3d', default=False, type=bool, help='Export the generated shapes and boxes in json files for future use')
parser.add_argument('--room_type', default='all', help='all, bedroom, livingroom, diningroom, library')

parser.add_argument(
        "--config_file",
        help="Path to the file that contains the experiment configuration",
        default="./Arrange/config/config_scene.yaml"
    )

parser.add_argument(
        "--ckpt",
        help="Path to the file that contains the experiment configuration",
        default="/remote-home/2332082/ArrangeBot/Arrange_15-0001-01-01 00:00:00/my_ddpm_model-0.1924.pt"
    )

args = parser.parse_args()

def reseed(num):
    np.random.seed(num)
    torch.manual_seed(num)
    random.seed(num)

def quaternion_to_angle(quaternion):
    """
    将四元数转换为角度值。
    
    参数:
    quaternion -- 形状为 (N, 4) 的张量，包含 N 个四元数，格式为 (w, x, y, z)
    
    返回:
    angles -- 形状为 (N,) 的张量，包含每个四元数对应的旋转角度（单位为度）
    """
    # 提取四元数的标量部分 w
    w = quaternion[:, 0]  # 选择第一列作为 w

    w = torch.clamp(w, -1.0, 1.0)
    # 计算旋转角度（弧度）
    theta = 2 * torch.acos(w)

    # 将弧度转换为角度（度）
    angles = theta * (180 / torch.pi)

    return angles

def validate_constrains_loop(img_path, test_dataset, model):
    print("开始加载数据")
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        collate_fn=test_dataset.collate_fn,
        shuffle=False,
        num_workers=1)
    print("数据加载完成，开始迭代...")
    # vocab = test_dataset.vocab


    for i, data in enumerate(tqdm(test_dataloader, desc=f"eval start",colour="#005500")):
        

        print(f'开始验证第{i+1}个场景')
        dec_objs, dec_triples = data['class_ids'], data['triples']
           
        
        name_dict=data["all_names"]
        dec_objs, dec_triples = dec_objs.cuda(), dec_triples.cuda()
        

        # all_pred_boxes = []
        # all_pred_angles = []

        with torch.no_grad():

            data_dict = model.generate_layout_sg(10,objs=dec_objs, 
                                                 triples=dec_triples)
            print(" loading layoutinfo")
            boxes_pred, angles_pred = torch.concat((data_dict['sizes'],data_dict['translations']),dim=-1), data_dict['angles']
            angles_pred=quaternion_to_angle(angles_pred).unsqueeze(1)
            print(angles_pred.shape)
            print(boxes_pred.shape)
            # if modelArgs['bin_angle']:
            #     angles_pred = -180 + (torch.argmax(angles_pred, dim=1, keepdim=True) + 1)* 15.0 # angle (previously minus 1, now add it back)
                 #将标准化的边界框参数转换回原始值
            boxes_pred_den = batch_torch_denormalize_box_params(boxes_pred) # mean, std
           
            # else:
            #     #将正余弦的弧度值转换为角度
            #     angles_pred = postprocess_sincos2arctan(angles_pred) / np.pi * 180
            #     #标准化的边界框参数转换回原始值
            #     boxes_pred_den = descale_box_params(boxes_pred, file=normalized_file) # min, max


        
        #classes = sorted(list(set(vocab['object_idx_to_name'])))
            # layout and shape visualization through open3d
        #print("rendering", [classes[i].strip('\n') for i in dec_objs])
        print("rendering obj ids:",dec_objs)
        render(name_dict,dec_objs.detach().cpu().numpy(), boxes_pred_den, angles_pred,
                store_path=img_path)
         

        # all_pred_boxes.append(boxes_pred.cpu().detach())
        # all_pred_angles.append(angles_pred.cpu().detach())
        # # compute constraints accuracy through simple geometric rules
        # accuracy = validate_constrains(dec_triples, boxes_pred, angles_pred, None, model.vocab, accuracy)

    # keys = list(accuracy.keys())
    # file_path_for_output = os.path.join(modelArgs['store_path'], f'{test_dataset.eval_type}_accuracy_analysis.txt')
    # with open(file_path_for_output, 'w') as file:
    #     for dic, typ in [(accuracy, "acc")]:
    #         lr_mean = np.mean([np.mean(dic[keys[0]]), np.mean(dic[keys[1]])])
    #         fb_mean = np.mean([np.mean(dic[keys[2]]), np.mean(dic[keys[3]])])
    #         bism_mean = np.mean([np.mean(dic[keys[4]]), np.mean(dic[keys[5]])])
    #         tash_mean = np.mean([np.mean(dic[keys[6]]), np.mean(dic[keys[7]])])
    #         stand_mean = np.mean(dic[keys[8]])
    #         close_mean = np.mean(dic[keys[9]])
    #         symm_mean = np.mean(dic[keys[10]])
    #         total_mean = np.mean(dic[keys[11]])
    #         means_of_mean = np.mean([lr_mean, fb_mean, bism_mean, tash_mean, stand_mean, close_mean, symm_mean])
    #         print(
    #             '{} & L/R: {:.2f} & F/B: {:.2f} & Bi/Sm: {:.2f} & Ta/Sh: {:.2f} & Stand: {:.2f} & Close: {:.2f} & Symm: {:.2f}. Total: &{:.2f}'.format(
    #                 typ, lr_mean,
    #                 fb_mean, bism_mean, tash_mean, stand_mean, close_mean, symm_mean, total_mean))
    #         print('means of mean: {:.2f}'.format(means_of_mean))
    #         file.write(
    #             '{} & L/R: {:.2f} & F/B: {:.2f} & Bi/Sm: {:.2f} & Ta/Sh: {:.2f} & Stand: {:.2f} & Close: {:.2f} & Symm: {:.2f}. Total: &{:.2f}\n'.format(
    #                 typ, lr_mean, fb_mean, bism_mean, tash_mean, stand_mean, close_mean, symm_mean, total_mean))
    #         file.write('means of mean: {:.2f}\n\n'.format(means_of_mean))


def evaluate():
    random.seed(48)
    torch.manual_seed(48)

    test_dataset_no_changes=my_Dataset(args.dataset,args.filepath,args.with_feats,args.if_debug)
 
    config=load_config(args.config_file)
   
    # args.visualize = False if args.gen_shape == False else args.visualize

    # instantiate the model
    
    
    model=DiffusionScene(config['network']).to(device=device)
    
    #model.optimizer_ini()
    check_point=torch.load(args.ckpt)
    model.load_state_dict(check_point,strict=True)
    print("模型已加载")
   

    model = model.eval()
 

  
    reseed(47)
    # validate_constrains_loop_w_changes(modelArgs, test_dataset_addition_changes, model, normalized_file=normalized_file, bin_angles=modelArgs['bin_angle'], cat2objs=cat2objs, datasize='large' if modelArgs['large'] else 'small', gen_shape=args.gen_shape)

    reseed(47)
    # print('\nEditing Mode - Relationship changes')
    # validate_constrains_loop_w_changes(modelArgs, test_dataset_rels_changes, model,  normalized_file=normalized_file, bin_angles=modelArgs['bin_angle'], cat2objs=cat2objs, datasize='large' if modelArgs['large'] else 'small', gen_shape=args.gen_shape)

    reseed(47)
    img_path='/remote-home/2332082/ArrangeBot/Arrange/scripts/train_iter'
    validate_constrains_loop(img_path, test_dataset_no_changes, model)

if __name__ == "__main__":
    print(torch.__version__)
    device=torch.device("cuda:0")
    evaluate()