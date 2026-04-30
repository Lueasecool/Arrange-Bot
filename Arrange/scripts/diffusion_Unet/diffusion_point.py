import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.distributions import Normal
import torch.distributed as dist
import math
import numpy as np
import torch.distributed as dist
from tqdm.auto import tqdm
import json
import torch.nn.functional as F
import sys
import os
sys.path.append('./SG-Bot/Arrange')
sys.path.append('./SG-Bot/Arrange/scripts')

#加入scene_ids
from diffusion_Unet.diffusion_gauss import GaussianDiffusion

def get_betas(schedule_type, b_start, b_end, time_num):
    if schedule_type == 'linear':
        betas = np.linspace(b_start, b_end, time_num)
    elif schedule_type == 'warm0.1':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.1)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == 'warm0.2':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.2)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == 'warm0.5':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.5)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == 'cosine':

        def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
            """
            Create a beta schedule that discretizes the given alpha_t_bar function,
            which defines the cumulative product of (1-beta) over time from t = [0,1].
            :param num_diffusion_timesteps: the number of betas to produce.
            :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                            produces the cumulative product of (1-beta) up to that
                            part of the diffusion process.
            :param max_beta: the maximum beta to use; use values lower than 1 to
                            prevent singularities.
            """
            betas = []
            for i in range(num_diffusion_timesteps):
                t1 = i / num_diffusion_timesteps
                t2 = (i + 1) / num_diffusion_timesteps
                betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
            
            return np.array(betas).astype(np.float64)
        
        betas_for_alpha_bar(
            time_num,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )

    else:
        raise NotImplementedError(schedule_type)
    return betas

class DiffusionPoint(nn.Module):
    def __init__(self, denoise_net, config, schedule_type='linear', beta_start=0.0001, beta_end=0.02, time_num=1000, 
            loss_type='mse', model_mean_type='eps', model_var_type ='fixedsmall', loss_separate=False, loss_iou=False):
          
        super(DiffusionPoint, self).__init__()
        
        betas = get_betas(schedule_type, beta_start, beta_end, time_num)
        
        self.diffusion = GaussianDiffusion(config, betas, loss_type, model_mean_type, model_var_type, loss_separate, loss_iou)

        self.model = denoise_net


    def prior_kl(self, x0):
        return self.diffusion._prior_bpd(x0)

    def all_kl(self, x0, condition, condition_cross, clip_denoised=True):
        total_bpd_b, vals_bt, prior_bpd_b, mse_bt =  self.diffusion.calc_bpd_loop(self._denoise, x0,  condition, condition_cross, clip_denoised)

        return {
            'total_bpd_b': total_bpd_b,
            'terms_bpd': vals_bt,
            'prior_bpd_b': prior_bpd_b,
            'mse_bt':mse_bt
        }


    def _denoise(self,data, obj_embed, triples, t, condition_cross):
        # B, D,N= data.shape
        # assert data.dtype == torch.float
        # assert t.shape == torch.Size([B]) and t.dtype == torch.int64

        # out = self.model(data, t, condition, condition_cross)#由data输入1d-Unet后的预测的噪声，与data形状一致
        
        # assert out.shape == torch.Size([B, D, N])
        # return out
    
        B, D = data.shape
        assert data.dtype == torch.float
        assert t.shape == torch.Size([B]) and t.dtype == torch.int64
        # data = data.unsqueeze(1)
        if self.model.conditioning_key == 'concat':
            out = self.model(data, obj_embed, triples, t)
        elif self.model.conditioning_key == 'crossattn':
            out = self.model(data, obj_embed, triples, t, context=condition_cross)
        else:
            raise NotImplementedError
        # elif self.model.conditioning_key == 'hybrid':
        #     out = self.model(data, condition, t, context=condition_cross)
        out = out.squeeze(-1)

       # assert out.shape == torch.Size([B, D])
        return out

    def get_loss_iter(self, data, noises=None, condition=None, condition_cross=None):
        
        if len(data.shape) == 3:
            B, D, N = data.shape
        elif len(data.shape) == 4:
            B, D, M, N = data.shape
        t = torch.randint(0, self.diffusion.num_timesteps, size=(B,), device=data.device)

        if noises is not None:
            noises[t!=0] = torch.randn((t!=0).sum(), *noises.shape[1:]).to(noises)#
#   为  noises  张量中与  t  中非零元素对应的位置生成随机噪声。生成的噪声张量的形状与  noises  张量相匹配
        losses, loss_dict = self.diffusion.p_losses(
            denoise_fn=self._denoise, data_start=data, t=t, noise=noises, condition=condition, condition_cross=condition_cross)
        assert losses.shape == t.shape == torch.Size([B])
        return losses.mean(), loss_dict
    
    def get_loss_iter_v2(self, obj_embed, preds, data,scene_ids, condition_cross=None):
        B, _ = data.shape
        unique_scenes, inv_idx = np.unique(scene_ids.cpu(), return_inverse=True)
        t = torch.randint(0, self.diffusion.num_timesteps, size=unique_scenes.shape,
                          device=data.device)  # we want to have different t for each scene not each obj
        t = t[inv_idx]
        assert len(t) == B
      

        loss = self.diffusion.p_losses(self._denoise, data, obj_embed, triples=preds, t=t, condition_cross=condition_cross)
        assert t.shape == torch.Size([B])
        return loss


    def gen_samples_sg(self, shape, device, obj_embed, triples=None, condition=None, noise_fn=torch.randn,
                    clip_denoised=True, keep_running=False):
        return self.diffusion.p_sample_loop_sg(self._denoise, shape=shape, device=device, obj_embed=obj_embed, triples=triples, condition=condition, noise_fn=noise_fn,
                                            clip_denoised=clip_denoised, keep_running=keep_running)