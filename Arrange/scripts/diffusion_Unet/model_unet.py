
import sys
sys.path.append('.')
sys.path.append('./scripts')
sys.path.append('./scripts/diffusion_Unet')
import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
from Arrange.scripts.diffusion_Unet.diffusion_point import DiffusionPoint
from diffusion_Unet.denoise_net import Unet1D
from diffusion_Unet.denoise_net_cross_attension import UNet1DModel
from diffusion_Unet.stat_logger import StatsLogger
from graph_net import *
class DiffusionScene(Module):

    def __init__(self, config,embedding_dim=128,gconv_pooling='avg', gconv_num_layers=5):
        super().__init__()
        self.text_condition=config.get("text_condition",False)
        self.diffusion=DiffusionPoint(denoise_net = UNet1DModel(**config["denoiser_kwargs"]),
            config = config,**config["diffusion_kwargs"]
            )   
        num_objs=14
        num_preds=16
        gconv_dim = embedding_dim
        self.obj_embeddings_ec = nn.Embedding(num_objs + 1, gconv_dim * 2)#num_objs 是该场景下物体个数
        self.pred_embeddings_ec = nn.Embedding(num_preds, gconv_dim * 2)#num_pres是物体关系总数
        self.config=config
        
        gconv_hidden_dim = gconv_dim * 4
        gconv_kwargs_ec = {
            'input_dim_obj': gconv_dim * 2 ,
            'input_dim_pred': gconv_dim * 2 ,
            'hidden_dim': gconv_hidden_dim,
            'pooling': gconv_pooling,
            'num_layers': gconv_num_layers,
            'mlp_normalization': None,
            'residual': False,
            'output_dim': gconv_dim * 2
        }
        self.gconv_net_enc=GraphTripleConvNet(**gconv_kwargs_ec)

    def encoder(self,objs,triples):
        # objs=datum["class_ids"]
        # triples=datum["triples"]
        # print(triples.shape)
        # bbox_es=datum['bbox_es']
        O, T = objs.size(0), triples.size(0)
        s, p, o = triples.chunk(3, dim=1)  # All have shape (T, 1)
        
        s, p, o = [x.squeeze(1) for x in [s, p, o]]  # Now have shape (T,)
        #print(T)
        edges = torch.stack([s, o], dim=1)  # Shape is (T, 2)
        # print("edges'shape:,",edges.shape)
        obj_embed = self.obj_embeddings_ec(objs)
        # print("obj_embding:",obj_embed.shape)
        pred_embed = self.pred_embeddings_ec(p)
        # print("pre_embed's shape:",pred_embed.shape)
        latent_obj_f, latent_pred_f = self.gconv_net_enc(obj_embed, pred_embed, edges)

        return obj_embed, pred_embed, latent_obj_f, latent_pred_f #obj_embed,latent_obj_f


    def get_loss(self,datum):

        class_ids=datum["class_ids"]
        class_ids_expanded = class_ids.unsqueeze(-1)  # 扩展一个维度，变为 [126, 7, 1]
        bbox_es=datum["bbox_es"]
        locations=datum["locations"]
        quaternion_xyzw=datum['quaternion_xyzw']  
        triples=datum['triples']
        scene_ids=datum["all_obj_to_scene"]
        # print("class_ids:", class_ids.shape)
        # print("bbox_es:", bbox_es.shape)
        # print("locations:", locations.shape)  # 这里可能会输出一个元组
        # print("quaternion_xyzw:", quaternion_xyzw.shape)
        layout_info=torch.cat([bbox_es,quaternion_xyzw],dim=-1).contiguous()
        obj_embed,_,latent_obj_f,_=self.encoder(class_ids,triples)
        self.loss = self.diffusion.get_loss_iter_v2(obj_embed=obj_embed, preds=triples, data=layout_info, scene_ids=scene_ids,condition_cross=latent_obj_f)
        return self.loss
        
        # loss,loss_dict=self.diffusion.get_loss_iter(
        #     layout_info,condition=None,condition_cross=None)
        
        # return loss,loss_dict
   
    def sample(self, box_dim, batch_size, objs=None, triples=None, clip_denoised=False):
        obj_embed,_,latent_obj_f,_=self.encoder(objs,triples)
        noise_shape = (batch_size, box_dim)
        condition = latent_obj_f
        condition_cross = None
        # reverse sampling
        samples = self.diffusion.gen_samples_sg(noise_shape, obj_embed.device, obj_embed, triples, condition=condition, clip_denoised=clip_denoised)
        
        return samples

    @torch.no_grad()
    def generate_layout_sg(self, box_dim, objs,triples,clip_denoised=False):
        samples = self.sample(box_dim, batch_size=len(objs), objs=objs, triples=triples, clip_denoised=clip_denoised)
        samples_dict = {
            "sizes": samples[:, 0:3].contiguous(),
            "translations": samples[:, 3:6].contiguous(),
            "angles": samples[:, 6:10].contiguous(),
        }
        
        return samples_dict
    
    def optimizer_ini(self):
        gcn_layout_df_params = [p for p in self.parameters() if p.requires_grad == True]
        shape_df_params = self.ShapeDiff.trainable_params
        trainable_params = gcn_layout_df_params + shape_df_params
        self.optimizerFULL = optim.AdamW(trainable_params, lr=1e-4)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizerFULL, lr_lambda=self.lr_lambda)
        self.optimizers = [self.optimizerFULL]
            
    

def train_on_batch(model, optimizer, sample_params, config):
    # Make sure that everything has the correct size
    optimizer.zero_grad()
    # Compute the loss
    loss, loss_dict = model.get_loss(sample_params)
    for k, v in loss_dict.items():
        StatsLogger.instance()[k].value = v.item()#遍历损失字典，记录每个损失项的值
    # Do the backpropagation
    print("loss:",loss.shape)
    loss.backward()
    # Compuite model norm
    grad_norm = clip_grad_norm_(model.parameters(), config["training"]["max_grad_norm"])#梯度裁剪
    StatsLogger.instance()["gradnorm"].value = grad_norm.item()#记录梯度范数
    # log learning rate
    StatsLogger.instance()["lr"].value = optimizer.param_groups[0]['lr']#记录学习率
    # Do the update
    optimizer.step()
    return loss.item()


def validate_on_batch(model, sample_params, config):
    # Compute the loss
    loss, loss_dict = model.get_loss(sample_params)
    for k, v in loss_dict.items():
        StatsLogger.instance()[k].value = v.item()
    return loss.item()





