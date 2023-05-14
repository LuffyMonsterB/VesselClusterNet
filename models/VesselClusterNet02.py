import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.layers import Conv, trunc_normal_
from monai.networks.blocks.transformerblock import TransformerBlock
import random
import monai.transforms as transforms
import math

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class PatchEmbeddingBlock(nn.Module):
    def __init__(self, in_channels: int,
                 image_size: int,
                 patch_size: int,
                 emb_dim: int,
                 num_heads: int,
                 dropout_rate: float = 0.0,
                 spatial_dims: int = 3,
                 ) -> None:
        super().__init__()
        assert image_size % patch_size == 0, 'image size must be divisible by patch size'
        assert emb_dim % num_heads == 0, 'embedding size should be divisible by num_heads.'

        num_patches = (image_size // patch_size) ** 3
        patch_dim = in_channels * patch_size ** 3
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches, emb_dim))
        self.dropout = nn.Dropout(dropout_rate)
        self.patch_embeddings = nn.Sequential(
            nn.Conv3d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(start_dim=2)
        )
        trunc_normal_(self.position_embeddings, mean=0.0, std=0.02, a=-2.0, b=2.0)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embeddings(x).transpose(-1, -2)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class ViT(nn.Module):
    def __init__(self, image_size=256, patch_size=8, in_channels=1, emb_dim=64, depth=8, heads=8,
                 mlp_dim=256, dropout=0.1):
        super().__init__()
        self.patch_embedding_block = PatchEmbeddingBlock(in_channels=in_channels, image_size=image_size,
                                                         patch_size=patch_size, emb_dim=emb_dim, num_heads=heads,
                                                         dropout_rate=dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size=emb_dim, mlp_dim=mlp_dim, num_heads=heads, dropout_rate=dropout,
                              qkv_bias=False) for i in range(depth)]
        )
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x):
        x = self.patch_embedding_block(x)
        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)
        # X: [B Patch C]
        return x, hidden_states_out

# todo 修改为填充+缩放的方式
class ReshapedTensor:
    def __init__(self, tensor):
        self.tensor = tensor
        self.orig_shape = tensor.shape[1:]
        self.new_shape = None

    def reshape(self, new_shape):
        self.new_shape = new_shape
        resize = transforms.Resize(new_shape)
        self.tensor = resize(self.tensor)
        return self.tensor

    def restore_orig_shape(self):
        c = self.tensor.shape[0]
        orig_shape = (c,) + self.orig_shape
        self.tensor = self.resize(orig_shape)(self.tensor)
        return self.tensor

    def restore_new_shape(self):
        c = self.tensor.shape[0]
        new_shape = (c,) + self.new_shape
        resize = transforms.Resize(new_shape)
        self.tensor = resize(self.tensor)
        return self.tensor

    def set_tensor(self, new_tensor):
        self.tensor = new_tensor
        return True


if __name__ == '__main__':
    img_size = (32, 32, 32)
    patch_size = 4
    vit = ViT(image_size=32, patch_size=4, in_channels=1, emb_dim=64, depth=4, heads=8, mlp_dim=256)
    num = 10
    datas = []
    inputs = []
    for i in range(num):
        data = torch.randn(1, random.randint(16, 48), random.randint(16, 48), random.randint(16, 48))
        tensor = ReshapedTensor(data)
        reshaped_tensor = tensor.reshape(img_size)
        datas.append(tensor)
        inputs.append(reshaped_tensor)
    inputs = torch.stack(inputs)
    outputs = vit(inputs)[0]

    outputs = outputs.detach().tolist()

    for i in range(len(outputs)):
        new_tensor = torch.tensor(outputs[i]).transpose(-1, -2).reshape(-1,img_size[0]//patch_size,img_size[1]//patch_size,img_size[2]//patch_size)
        datas[i].set_tensor(new_tensor)
        datas[i].restore_new_shape()
        datas[i].restore_orig_shape()

    print(datas)
