import monai
import torch
import numpy as np
import SimpleITK as sitk
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

# 使用sitk加载nii数据
def load_data(path):
    nii = sitk.ReadImage(path)
    # D H W
    arr = sitk.GetArrayFromImage(nii)
    return arr


def make_coord(shape, ranges=None, flatten=True):
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

def visualize(coord, colors):

    # 获取数组的坐标轴信息
    z, y, x = coord[:,0],coord[:,1],coord[:,2]
    # 创建一个3D图像对象
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # 绘制散点图
    ax.scatter(x, y, z, c=colors, s=1)
    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # 显示图像
    plt.show()

# 加载nii为数组
nii_path = '../Data/Parse22_seg/val/seg/PA000282.nii.gz'
nii_arr = load_data(nii_path)
nii_arr[nii_arr>0.5]=1
nii_arr[nii_arr<=0.5]=0
# 建立标准坐标系
coord = make_coord(shape=nii_arr.shape, flatten=False)
# 筛选前景区域的坐标
coord_nonzero = coord[nii_arr > 0].numpy()

num_cluster = 1
kmeans_list = KMeans(n_clusters=num_cluster, max_iter=30).fit(coord_nonzero)
center_list =kmeans_list.labels_
center_index = np.unique(center_list)
# 可视化
color_map = {}
for idx in center_index:
    color_map[idx]=np.random.randint(0, 255, size=(3))/255

colors = []
for i in center_list:
    colors.append(color_map[i])
visualize(coord_nonzero,colors)
