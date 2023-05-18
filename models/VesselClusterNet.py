import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import monai.transforms as transforms
import math
from sklearn.cluster import KMeans
from .SegResNet3D import SegResNet
from .ViT import PatchEmbeddingBlock, ViT
from .utils import fea_to_binary, make_coord, min_bounding_box, ReshapedTensor3D
import math
import numpy as np


class VesselClusterNet(nn.Module):
    def __init__(self,num_cluster = 20):
        super().__init__()
        self.img_net = SegResNet(
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=16,
            in_channels=1,
            out_channels=1,
            dropout_prob=0.1,
        )
        self.img_net.load_state_dict(torch.load('D:\Projects\llf\VesselClusterNet\checkpoints\segresnet\epoch-190_dice-0.8242.pth')['state_dict'],strict=False)
        self.patch_format_size =(16,16,16)
        self.num_cluster = num_cluster
        self.vit = ViT(image_size=self.patch_format_size[0], patch_size=4,num_cluster = num_cluster, in_channels=16, emb_dim=64, depth=4, heads=8, mlp_dim=256)

    # def _make_vit_layers(self):
    #     vit_layers = nn.ModuleList()
    #     num_layers = 3
    #     for i in range(num_layers):
    #         vit_layers.append(
    #             ViT(image_size=32, patch_size=4, in_channels=1, emb_dim=64, depth=4, heads=8, mlp_dim=256))
    #     return vit_layers

    def gen_patch_by_cluster(self, fea, mask):
        b, c, d, h, w = fea.shape
        b_patch_feas = []
        # b_cluster_coords = []
        b_bounding_boxs = []
        # 建立标准坐标系
        coord = make_coord(shape=mask.shape[-3:], flatten=False).unsqueeze(0)
        for i in range(b):
            # 聚类生成图块, 筛选前景区域的坐标
            coord_nonzero = coord[mask[i, :, :, :, :] > 0].numpy()
            kmeans_list = KMeans(n_clusters=self.num_cluster, max_iter=30).fit(coord_nonzero)
            label_list = kmeans_list.labels_
            cluster_list = [[] for _ in range(self.num_cluster)]
            # 按簇对点进行分类
            for idx, label in enumerate(label_list):
                cluster_list[label].append(coord_nonzero[idx, :])
            # batch内进行图块抽取
            bounding_box_list = []
            patch_fea_list = []
            for c in range(self.num_cluster):
                # 获取外包立方体最小点和最大点下标
                bounding_box = min_bounding_box(cluster_list[c])
                truth_max_point = (bounding_box['max_point'] + 1) * (d, h, w) / 2
                truth_max_point = [math.floor(p) for p in truth_max_point]
                truth_min_point = (bounding_box['min_point'] + 1) * (d, h, w) / 2
                truth_min_point = [math.floor(p) for p in truth_min_point]
                print(truth_max_point,truth_min_point)
                # 抽取语义特征
                patch_fea_list.append(
                    fea[i, :, truth_min_point[0]:truth_max_point[0], truth_min_point[1]:truth_max_point[1],
                    truth_min_point[2]:truth_max_point[2]])
                bounding_box_list.append({"max_point": truth_max_point, "min_point": truth_min_point})

            # b_cluster_coords.append(cluster_list)
            b_bounding_boxs.append(bounding_box_list)
            b_patch_feas.append(patch_fea_list)
        return b_patch_feas, b_bounding_boxs

    def forward(self, x):
        coarse_output = self.img_net(x)
        coarse_seg = coarse_output['output']

        down_fea = coarse_output["down_fea"]
        up_outputs = coarse_output["up_outputs"]
        up_outputs = fea_to_binary(up_outputs)
        with torch.no_grad():
            mask = up_outputs[-1].clone()
            fea = down_fea[-1].clone()
            b_patch_feas, b_bounding_boxs = self.gen_patch_by_cluster(fea, mask)
            b_patch_reshaped_feas = []
            b_patch_format_feas = []
            for b in b_patch_feas:
                reshaped_fea_list = []
                format_fea_list = []
                for patch in b:
                    reshaped_fea = ReshapedTensor3D(patch)
                    reshaped_fea_list.append(reshaped_fea)
                    format_fea_list.append(reshaped_fea.reshape(self.patch_format_size))
                b_patch_reshaped_feas.append(reshaped_fea_list)
                b_patch_format_feas.append(torch.stack(format_fea_list))
            # shape: B Num_patch C D H W
        format_fea = torch.stack(b_patch_format_feas).clone().detach()
        vit_out = self.vit(format_fea)
        fine_outs = []

        for b in range(len(b_patch_reshaped_feas)):
            reshaped_outs = []
            for i,fea in enumerate(b_patch_reshaped_feas[b]):
                fea.restore(updated_tensor = vit_out[b,i,:,:,:].unsqueeze(0))
                reshaped_outs.append(fea.tensor)
            fine_outs.append(reshaped_outs)

        # todo: reshape tensor
        return coarse_seg, fine_outs,b_bounding_boxs


if __name__ == '__main__':
    vessel_cluster_net = VesselClusterNet()
    data = torch.randn(2, 1, 32, 64, 64)
    coarse_seg, down_fea, up_outputs = vessel_cluster_net(data)
    print(coarse_seg.shape, down_fea, up_fea)

# if __name__ == '__main__':
#     img_size = (32, 32, 32)
#     patch_size = 4
#     vit = ViT(image_size=32, patch_size=4, in_channels=1, emb_dim=64, depth=4, heads=8, mlp_dim=256)
#     num = 10
#     datas = []
#     inputs = []
#     for i in range(num):
#         data = torch.randn(1, random.randint(16, 48), random.randint(16, 48), random.randint(16, 48))
#         tensor = ReshapedTensor(data)
#         reshaped_tensor = tensor.reshape(img_size)
#         datas.append(tensor)
#         inputs.append(reshaped_tensor)
#     inputs = torch.stack(inputs)
#     outputs = vit(inputs)[0]
#
#     outputs = outputs.detach().tolist()
#
#     for i in range(len(outputs)):
#         new_tensor = torch.tensor(outputs[i]).transpose(-1, -2).reshape(-1, img_size[0] // patch_size,
#                                                                         img_size[1] // patch_size,
#                                                                         img_size[2] // patch_size)
#         datas[i].set_tensor(new_tensor)
#         datas[i].restore_new_shape()
#         datas[i].restore_orig_shape()
#
#     print(datas)
