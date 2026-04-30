from __future__ import print_function

import open3d as o3d # open3d needs to be imported before other packages!
import argparse
import os
import random
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data

import sys
from pathlib import Path


def batch_torch_destandardize_box_params(box_params, file=None, scale=3, params=7):
    """ Denormalize the box parameters utilizing the accumulated dataset statistics

    :param box_params: float tensor of shape [N, 6] containing the 6 box parameters, where N is the number of boxes
    :param scale: float scalar that scales the parameter distribution
    :return: float tensor of shape [N, 6], the denormalized box parameters
    """
    assert file is not None
    if file == None:
        if params == 6:
            mean = torch.from_numpy(np.array([1.3827214, 1.309359, 0.9488993, -0.12464812, 0.6188591, -0.54847]).reshape(1,-1)).cuda()
            std = torch.from_numpy(np.array([1.7797655, 1.657638, 0.8501885, 1.9160025, 2.0038228, 0.70099753]).reshape(1,-1)).cuda()
        elif params == 7:
            mean = torch.from_numpy(np.array([1.3827214, 1.309359, 0.9488993, -0.12464812, 0.6188591, -0.54847, 0.73127955]).reshape(1,-1)).cuda()
            std = torch.from_numpy(np.array([1.7797655, 1.657638, 0.8501885, 1.9160025, 2.0038228, 0.70099753, 0.50347435]).reshape(1,-1)).cuda()
        else:
            raise NotImplementedError
    else:
        stats = np.loadtxt(file)
        if params == 6:
            mean, std = torch.from_numpy(stats[0][:6].reshape(1,-1)).cuda(), torch.from_numpy(stats[1][:6].reshape(1,-1)).cuda()
        else:
            mean, std = torch.from_numpy(stats[0].reshape(1,-1)).cuda(), torch.from_numpy(stats[1].reshape(1,-1)).cuda()

    return (box_params * std) / scale + mean



def postprocess_sincos2arctan(sincos):
    if isinstance(sincos, np.ndarray):
        assert sincos.shape[1] == 2
        return np.arctan2(sincos[0],sincos[1])
    elif isinstance(sincos,torch.Tensor):
        B, N = sincos.shape
        assert N == 2
        return torch.arctan2(sincos[:,0], sincos[:,1]).reshape(B,1)
    else:
        raise NotImplementedError


def descale_box_params(normed_box_params, file=None, angle=False):
    assert file is not None
    stats = np.loadtxt(file)
    if isinstance(normed_box_params,torch.Tensor):
        stats = torch.tensor(stats,dtype=normed_box_params.dtype, device=normed_box_params.device)
    min_lhw, max_lhw, min_xyz, max_xyz, min_angle, max_angle = stats[:3], stats[3:6], stats[6:9], stats[9:12], stats[12:13], stats[13:]
    normed_box_params[:,:3] = (normed_box_params[:,:3] + 1) / 2
    normed_box_params[:,:3] = normed_box_params[:,:3] * (max_lhw - min_lhw) + min_lhw # size

    normed_box_params[:, 3:6] = (normed_box_params[:, 3:6] + 1) / 2
    normed_box_params[:,3:6] = normed_box_params[:,3:6] * (max_xyz - min_xyz) + min_xyz # loc
    if angle:
        normed_box_params[:,6:7] = (normed_box_params[:,6:7] + 1) / 2
        normed_box_params[:,6:7] = normed_box_params[:,6:7] * (max_angle - min_angle) + min_angle  # angle

    return normed_box_params