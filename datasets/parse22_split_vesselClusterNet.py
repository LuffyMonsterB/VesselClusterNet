from monai import transforms
import os
from glob import glob
from torch.utils.data import Dataset, DataLoader
import torch
from einops import rearrange
from sklearn.cluster import KMeans
import numpy as np
import math

class Parse22NiiSplitDataset(Dataset):
    def __init__(self, nii_files, label_files, trans, patch_size, num_cluster):
        super().__init__()
        self.nii_files = nii_files
        self.label_files = label_files
        self.trans = trans
        self.patch_size = patch_size
        self.num_cluster = num_cluster

    def __len__(self):
        return len(self.nii_files)

    def normalize(self, data):
        hu_max = 512
        hu_min = -968
        data[data > hu_max] = hu_max
        data[data < hu_min] = hu_min
        data = (data - (hu_min)) / (hu_max - hu_min)
        data = torch.FloatTensor(data)
        return data

    def make_coord(self, shape, ranges=None, flatten=True):
        coord_seqs = []
        for i, n in enumerate(shape):
            if ranges is None:
                v0, v1 = -1, 1
            else:
                # v0, v1 = ranges[i]
                v0, v1 = ranges
            r = (v1 - v0) / (2 * n)
            seq = v0 + r + (2 * r) * torch.arange(n).float()
            coord_seqs.append(seq)
        ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
        if flatten:
            ret = ret.view(-1, ret.shape[-1])
        return ret

    # 获取外包立方体
    # 未使用旋转卡壳算法，因为只需要水平垂直方向上外包即可
    def min_bounding_box(self, points):
        points = np.array(points)
        # 找到最小包围盒的最小点和最大点
        min_point = np.min(points, axis=0)
        max_point = np.max(points, axis=0)

        return {"min_point": min_point, "max_point": max_point}

    def gen_patch_by_cluster(self, label):
        c, d, h, w = label.shape
        bounding_boxs = []
        # 建立标准坐标系
        coord = self.make_coord(shape=[d, h, w], flatten=False).unsqueeze(0)

        # 聚类生成图块, 筛选前景区域的坐标
        coord_nonzero = coord[label > 0].numpy()
        kmeans_list = KMeans(n_clusters=self.num_cluster, max_iter=30).fit(coord_nonzero)
        label_list = kmeans_list.labels_
        cluster_list = [[] for _ in range(self.num_cluster)]
        # 按簇对点进行分类
        for idx, label_ in enumerate(label_list):
            cluster_list[label_].append(coord_nonzero[idx, :])
        for c in range(self.num_cluster):
            # 获取外包立方体最小点和最大点下标
            bounding_box = self.min_bounding_box(cluster_list[c])
            truth_max_point = (bounding_box['max_point'] + 1) * (d, h, w) / 2
            truth_max_point = [math.floor(p) for p in truth_max_point]
            truth_min_point = (bounding_box['min_point'] + 1) * (d, h, w) / 2
            truth_min_point = [math.floor(p) for p in truth_min_point]

            divisible = True
            # 适配patch size
            for i in range(3):
                l = truth_max_point[i] - truth_min_point[i]
                if l % self.patch_size != 0:
                    diff = self.patch_size - l % self.patch_size
                    if truth_max_point[i] + diff < label.shape[i + 1]:
                        truth_max_point[i] = truth_max_point[i] + diff
                    elif truth_min_point[i] - diff >= 0:
                        truth_min_point[i] = truth_min_point[i] - diff
                    else:
                        divisible = False
            if divisible:
                bounding_boxs.append({"max_point": truth_max_point, "min_point": truth_min_point})
        return bounding_boxs

    def __getitem__(self, item):
        nii_file = self.nii_files[item]
        label_file = self.label_files[item]
        file = {'image': nii_file, 'label': label_file}
        file = self.trans(file)
        nii = file['image']
        nii = rearrange(nii, "C W H D -> C D H W")
        label = file['label']
        label = rearrange(label, "C W H D -> C D H W")

        bounding_boxs = self.gen_patch_by_cluster(label)
        # 提取图块和块标签
        nii_patches = []
        label_patches = []
        for bounding_box in bounding_boxs:
            max_point, min_point = bounding_box['max_point'], bounding_box['min_point']
            nii_patch = nii[:, min_point[0]:max_point[0], min_point[1]:max_point[1],
                        min_point[2]:max_point[2]]
            label_patch = label[:, min_point[0]:max_point[0], min_point[1]:max_point[1],
                          min_point[2]:max_point[2]]
            nii_patches.append(nii_patch)
            label_patches.append(label_patch)
        return nii_patches, label_patches


def load_parse22_data(data_dir):
    files_dir = sorted(glob(os.path.join(data_dir, "*")))
    niis = []
    labels = []
    for file_dir in files_dir:
        nii_dir = os.path.join(file_dir, "image")
        label_dir = os.path.join(file_dir, "label")
        niis.append(glob(os.path.join(nii_dir, "*.nii.gz")))
        labels.append(glob(os.path.join(label_dir, "*.nii.gz")))
    return niis, labels


def get_train_dataloader(data_dir, batch_size, num_workers,patch_size, num_cluster):
    train_niis, train_labels = load_parse22_data(data_dir)
    train_trans = transforms.Compose([
        transforms.LoadImageD(keys=['image', 'label']),
        transforms.EnsureChannelFirstD(keys=['image', 'label']),
        # transforms.RandSpatialCropD(keys=['image', 'label'], roi_size=(128, 128, 64), random_size=False),
        transforms.ScaleIntensityRangeD(keys=['image'], a_min=-968, a_max=512, b_min=0, b_max=1),
        transforms.EnsureTypeD(keys=['image', 'label'])
    ])

    train_ds = Parse22NiiSplitDataset(train_niis, train_labels, train_trans,patch_size, num_cluster)

    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                          pin_memory=torch.cuda.is_available())
    return train_dl


def get_val_dataloader(val_dir, val_batch_size, val_numworkers):
    val_niis, val_labels = load_parse22_data(val_dir)
    val_trans = transforms.Compose([
        transforms.LoadImageD(keys=['image', 'label']),
        transforms.EnsureChannelFirstD(keys=['image', 'label']),
        transforms.ScaleIntensityRangeD(keys=['image'], a_min=-968, a_max=512, b_min=0, b_max=1),
        transforms.EnsureTypeD(keys=['image', 'label'])
    ])

    val_ds = Parse22NiiDataset(val_niis, val_labels, val_trans)

    val_dl = DataLoader(val_ds, batch_size=val_batch_size, num_workers=val_numworkers, shuffle=True,
                        pin_memory=torch.cuda.is_available())

    return val_dl

if __name__ == '__main__':
    data_dir = '../../Data/Parse22/nii/train'
    dl = get_train_dataloader(data_dir,1,0,patch_size=4,num_cluster=20)
    for batch_data in dl:
        nii_patches,label_patches = batch_data

