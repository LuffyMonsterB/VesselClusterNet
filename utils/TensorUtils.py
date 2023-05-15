import torch
import torch.nn.functional as F

# todo: 编写说明文档
# 可变尺寸3D张量，在改变尺寸后提供恢复原始尺寸的功能
class ReshapedTensor3D:
    def __init__(self, tensor):
        self.tensor = tensor
        self.ori_shape = tensor.shape[-3:]
        self.new_shape = None
        self.scale = 1
        self.pad = {}

    # 使用填充+缩放进行reshape， 不改变通道数
    def reshape(self, new_shape):
        d, h, w = new_shape
        ori_d, ori_h, ori_w = self.ori_shape
        # 1. 根据最长边进行缩放
        scale = min(d / ori_d, h / ori_h, w / ori_w)
        new_tensor = F.interpolate(self.tensor, scale_factor=scale)
        self.scale = scale
        # 2. 对剩余边进行pad填充
        pad_d = (d - new_tensor.shape[-3] % d) if d != new_tensor.shape[-3] else 0
        pad_h = (h - new_tensor.shape[-2] % h) if w != new_tensor.shape[-2] else 0
        pad_w = (w - new_tensor.shape[-1] % w) if h != new_tensor.shape[-1] else 0
        # F.pad模式可查看：https://zhuanlan.zhihu.com/p/358599463
        # pad = (左边填充数， 右边填充数， 上边填充数， 下边填充数， 前边填充数，后边填充数)
        new_tensor = F.pad(new_tensor, (pad_w // 2, pad_w // 2 if pad_w % 2 == 0 else pad_w // 2 + 1,
                                        pad_h // 2, pad_h // 2 if pad_h % 2 == 0 else pad_h // 2 + 1,
                                        pad_d // 2, pad_d // 2 if pad_d % 2 == 0 else pad_d // 2 + 1))

        self.tensor = new_tensor
        self.new_shape = new_shape
        self.pad = {"d": pad_d, "h": pad_h, "w": pad_w}
        return self.tensor

    # 接收更新过的tensor，恢复张量尺寸
    def restore(self, updated_tensor=None):
        if updated_tensor is not None:
            self.tensor = updated_tensor
        # 1. 取消填充
        pad_d, pad_h, pad_w = self.pad['d'], self.pad['h'], self.pad['w']
        new_tensor = self.tensor[:, :, (pad_d // 2):(32 - pad_d // 2), (pad_h // 2):(32 - pad_h // 2), (pad_w // 2):(32 - pad_w // 2)]
        new_tensor = F.interpolate(new_tensor, self.ori_shape)
        self.tensor = new_tensor
        return self.tensor

if __name__ == '__main__':
    tensor = ReshapedTensor3D(torch.randn(2, 16, 68, 64, 64))
    reshaped_tensor = tensor.reshape(new_shape=(32, 32, 32))
    restored_tensor = tensor.restore(reshaped_tensor)
    print(reshaped_tensor.shape)
    print(restored_tensor.shape)