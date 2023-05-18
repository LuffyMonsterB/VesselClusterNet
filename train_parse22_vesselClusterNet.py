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
from datasets.parse22_vesselClusterNet import get_train_dataloader
from models.VesselClusterNet import VesselClusterNet


def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    checkpoint_dir = './checkpoints/vessel_cluster_net'
    # 加载dataloader
    train_dir = '../data/Parse22/train'
    train_dl = get_train_dataloader(data_dir=train_dir, batch_size=2, num_workers=0)
    # 定义loss函数和模型训练配置
    dice_loss = DiceCELoss(sigmoid=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VesselClusterNet(num_cluster = 16).to(device)
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
    for idx, batch_data in train_loop:
        nii, labels = batch_data[0].to(device), batch_data[1].to(device)
        for param in model.parameters():
            param.grad = None
        coarse_seg, fine_outs, b_bounding_boxs = model(nii)
        loss1,loss2 = cal_total_loss(loss_func, labels, coarse_seg, fine_outs, b_bounding_boxs)
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()
        run_loss.update(loss.item(), n=1)
        avg_patch_size = np.array([0,0,0])
        for b in range(len(fine_outs)):
            for p in fine_outs[b]:
                d,h,w = p.shape
                avg_patch_size += [d,h,w]
        avg_patch_size = avg_patch_size // (len(fine_outs)*len(fine_outs[0]))
        train_loop.set_postfix(train_avg_loss=round(np.mean(run_loss.avg), 4),loss2 = round(loss2.item(),4),avg_patch_size=avg_patch_size)
    for param in model.parameters():
        param.grad = None
    return run_loss.avg


# 计算总的损失，粗分割+细化输出
def cal_total_loss(loss_func, labels, coarse_seg, fine_out, b_bounding_boxs):
    loss_func2 = torch.nn.MSELoss()
    b_label_patches = get_label_patch(labels, b_bounding_boxs)
    loss1 = loss_func(coarse_seg, labels)
    loss2 = 0
    loss2_count = 0
    for b in range(len(b_label_patches)):
        for i,(out_patch,label_patch) in enumerate(zip(fine_out[b],b_label_patches[b])):
            loss_patch = loss_func2(label_patch,out_patch.unsqueeze(0))
            loss2 += loss_patch
            loss2_count += 1
    loss2 = loss2 / len(b_label_patches)
    return loss1 , loss2


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
