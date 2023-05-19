import logging
import os
import sys
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import monai
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose, EnsureType, Activations
from monai.utils.enums import MetricReduction
from monai.losses import DiceCELoss
from tqdm import tqdm
from utils.TrainUtils import AverageMeter, get_label_patch
from datasets.parse22_split_vesselClusterNet import get_train_dataloader
from models.ViT2 import ViT


def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    checkpoint_dir = './checkpoints/split_vessel_cluster_net'
    # 加载dataloader
    train_dir = '../data/Parse22/train'
    patch_size = 8
    train_dl = get_train_dataloader(data_dir=train_dir, batch_size=1, num_workers=0,patch_size=patch_size,num_cluster=30)
    # 定义loss函数和模型训练配置
    dice_loss = DiceCELoss(sigmoid=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViT(patch_size=patch_size).to(device)
    num_epoch = 400
    learn_rate = 5e-4
    optimizer = torch.optim.Adam(model.parameters(), learn_rate, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch)
    save_interval = 40
    writer = SummaryWriter()
    for epoch in range(num_epoch):
        train_loss = train_epoch(epoch, model, train_dl, optimizer, dice_loss, device)
        writer.add_scalar("train_loss", train_loss, epoch)
        scheduler.step()

        if (epoch + 1) % save_interval == 0:
            if not os.path.exists(checkpoint_dir):
                os.mkdir(checkpoint_dir)
            save_checkpoint(checkpoint_dir, model, epoch + 1, optimizer, scheduler)

    print(f"train completed")
    writer.close()


def train_epoch(epoch, model, dataloader, optimizer, loss_func, device):
    model.train()
    run_loss = AverageMeter()
    train_loop = tqdm(enumerate(dataloader), total=len(dataloader))
    train_loop.set_description(f'Epoch {epoch + 1}')
    for idx, (nii_patches, label_patches) in train_loop:
        for patch_data in zip(nii_patches,label_patches):
            for param in model.parameters():
                param.grad = None
            nii_patch, label_patch = patch_data[0].to(device),patch_data[1].to(device)

            output = model(nii_patch)
            loss = loss_func(output,label_patch)
            loss.backward()
            optimizer.step()
            run_loss.update(loss.item(), n=1)
        train_loop.set_postfix(train_avg_loss=round(np.mean(run_loss.avg), 4))
    return run_loss.avg

def save_checkpoint(save_dir, model, epoch, optimizer=None, scheduler=None):
    state_dict = model.state_dict()
    save_dict = {"epoch": epoch, "state_dict": state_dict}
    if optimizer is not None:
        save_dict['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        save_dict['scheduler'] = scheduler.state_dict()
    file_name = os.path.join(save_dir, f"epoch-{epoch}.pth")
    torch.save(save_dict, file_name)
    print(f"saving new checkpoint {file_name}")


if __name__ == "__main__":
    main()
