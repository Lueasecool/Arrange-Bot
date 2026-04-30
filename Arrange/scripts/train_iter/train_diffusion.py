
import argparse
import logging
import os
import sys
from tqdm.auto import tqdm
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import onnx
import onnx.utils
import onnx.version_converter
import time
from datetime import datetime

sys.path.append('./Arrange')
sys.path.append('./Arrange/scripts')
from scripts.train_iter.utils import load_config

from __init__ import optimizer_factory,schedule_factory,adjust_learning_rate

from scripts.diffusion_Unet.model_unet import train_on_batch
from diffusion_Unet.stat_logger import StatsLogger
from data_util.data_readfull import my_Dataset
from scripts.diffusion_Unet.model_unet import DiffusionScene
from diffusion_Unet.denoise_net import Unet1D
from diffusion_Unet.diffusion_gauss import GaussianDiffusion
from diffusion_Unet.diffusion_point import DiffusionPoint
def main(args):
    
    now = datetime.now()
    folder_name = f"Arrange_{now.hour}-{now.mi}"

# 创建文件夹
    os.makedirs(folder_name, exist_ok=True)



    np.random.seed(args.seed)
    torch.manual_seed(np.random.randint(np.iinfo(np.int32).max))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(np.random.randint(np.iinfo(np.int32).max))


    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(np.random.randint(np.iinfo(np.int32).max))

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    # device = torch.device("cpu")
    print("Running code on", device)

      
    config=load_config(args.config_file)

    train_dataset=my_Dataset(config['root'].get('raw'),config['root'].get('file_path'),None,None)
    

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"].get("batch_size", 128),
        num_workers=args.n_processes,
        collate_fn=train_dataset.collate_fn,
        shuffle=True
    )
  
    network=DiffusionScene(config['network']).to(device=device)
    

    
    
    optimizer = optimizer_factory(config["training"],
                                   filter(lambda p: p.requires_grad, network.parameters()) )
    lr_scheduler = schedule_factory(config["training"])
    num_epochs=config["training"].get("epochs", 5000)
    best_loss=100
    
    for epoch in tqdm(range(num_epochs),desc=f"Training progress",colour="#00ff00"):
        # adjust learning rate
        adjust_learning_rate(lr_scheduler, optimizer, epoch)
        epoch_loss=0.0
        log_string = f"Loss at epoch {epoch + 1}: {best_loss:.3f}"
        
        network.train()
        #for b, sample in zip(range(steps_per_epoch), yield_forever(train_loader)):
        for b, sample in enumerate(tqdm(train_loader, leave=False, desc=f"Epoch {epoch + 1}/{num_epochs}",colour="#005500")):
            #Move everything to device
            for k, v in sample.items():
                if not isinstance(v, list) and not isinstance(v, dict) :#只有当 v 不是列表,才将值调到(devcie)
                    sample[k] = v.to(device)
            batch_loss = train_on_batch(network, optimizer, sample, config)
            print('loss:',batch_loss)
            StatsLogger.instance().print_progress(epoch+1, b+1, batch_loss)
            
            epoch_loss += batch_loss
        if epoch%100==0:
            if best_loss > epoch_loss :
                best_loss = epoch_loss
                torch.save(network.state_dict(), os.path.join(folder_name,f"my_ddpm_model-{best_loss:.4f}.pt"))
                log_string += " --> Best model ever (stored)"
                print(log_string)
        StatsLogger.instance().clear()

       

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Train a generative model on bounding boxes"
    )

    parser.add_argument(
        "--config_file",
        help="Path to the file that contains the experiment configuration",
        default="./Arrange/config/config_scene.yaml"
    )
    parser.add_argument(
        "--weight_file",
        default=None,
        help=("The path to a previously trained model to continue"
              " the training from")
    )
    parser.add_argument(
        "--continue_from_epoch",
        default=0,
        type=int,
        help="Continue training from epoch (default=0)"
    )
    parser.add_argument(
        "--n_processes",
        type=int,
        default=0,
        help="The number of processed spawned by the batch provider"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=27,
        help="Seed for the PRNG"
    )
    parser.add_argument(
        "--experiment_tag",
        default=None,
        help="Tag that refers to the current experiment"
    )
    parser.add_argument(
        "--with_wandb_logger",
        action="store_true",
        help="Use wandB for logging the training progress"
    )

    args = parser.parse_args()
    main(args)    