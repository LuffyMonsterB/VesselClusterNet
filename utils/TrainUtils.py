import numpy as np
import torch

import sys

# 测试需要解开注释
# sys.path.append('../models')
# from VesselClusterNet import VesselClusterNet


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def get_label_patch(labels, bounding_box_list):
    b_label_patches = []
    for b in range(len(bounding_box_list)):
        label_patch_list = []
        for bounding_box in bounding_box_list[b]:
            max_point = bounding_box['max_point']
            min_point = bounding_box['min_point']
            label_patch = labels[b, :, min_point[0]:max_point[0], min_point[1]:max_point[1], min_point[2]:max_point[2]]
            label_patch_list.append(label_patch)
        b_label_patches.append(label_patch_list)
    return b_label_patches


if __name__ == '__main__':
    vessel_cluster_net = VesselClusterNet()
    data = torch.randn(2, 1, 32, 64, 64)
    labels = torch.randn(2, 1, 32, 64, 64)
    coarse_seg, fine_out, b_bounding_boxs = vessel_cluster_net(data)
    b_label_patches = get_label_patch(labels,b_bounding_boxs)
    print(coarse_seg, fine_out, b_bounding_boxs)
