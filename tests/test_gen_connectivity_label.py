import monai
import torch
import numpy as np
import SimpleITK as sitk

directions = [
    [1, 0, 0],  # 正 x 轴方向
    [-1, 0, 0],  # 负 x 轴方向
    [0, 1, 0],  # 正 y 轴方向
    [0, -1, 0],  # 负 y 轴方向
    [0, 0, 1],  # 正 z 轴方向
    [0, 0, -1],  # 负 z 轴方向
    [1, 1, 0],  # 正 x 正 y 方向
    [1, -1, 0],  # 正 x 负 y 方向
    [-1, 1, 0],  # 负 x 正 y 方向
    [-1, -1, 0],  # 负 x 负 y 方向
    [1, 0, 1],  # 正 x 正 z 方向
    [1, 0, -1],  # 正 x 负 z 方向
    [-1, 0, 1],  # 负 x 正 z 方向
    [-1, 0, -1],  # 负 x 负 z 方向
    [0, 1, 1],  # 正 y 正 z 方向
    [0, 1, -1],  # 正 y 负 z 方向
    [0, -1, 1],  # 负 y 正 z 方向
    [0, -1, -1],  # 负 y 负 z 方向
    [1, 1, 1],  # 正 x 正 y 正 z 方向
    [1, 1, -1],  # 正 x 正 y 负 z 方向
    [1, -1, 1],  # 正 x 负 y 正 z 方向
    [1, -1, -1],  # 正 x 负 y 负 z 方向
    [-1, 1, 1],  # 负 x 正 y 正 z 方向
    [-1, 1, -1],  # 负 x 正 y 负 z 方向
    [-1, -1, 1],  # 负 x 负 y 正 z 方向
    [-1, -1, -1],  # 负 x 负 y 负 z 方向
]


# 使用sitk加载nii数据
def load_data(path):
    nii = sitk.ReadImage(path)
    # D H W
    arr = sitk.GetArrayFromImage(nii)
    return arr


# 加载nii为数组
label_path = '../../Data/Parse22_seg/val/label/PA000282.nii.gz'
label = load_data(label_path)
d, h, w = label.shape
label_connectivity = np.zeros([d, h, w, 26])
for i in range(d):
    for j in range(h):
        for k in range(w):
            if label[i, j, k] != 0:
                for direction, (dd, dh, dw) in enumerate(directions):
                    if 0 <= i + dd < d and 0 <= j + dh < h and 0 <= k + dw < w:
                        if label[i + dd, j + dh, k + dw] == 1:
                            label_connectivity[i, j, k, direction] = 1
np.save('label_connectivity.npy',label_connectivity)
print(label_connectivity)
