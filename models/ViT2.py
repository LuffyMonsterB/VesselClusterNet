from monai.networks.blocks.transformerblock import TransformerBlock
from monai.networks.layers import Conv, trunc_normal_
import torch
import torch.nn as nn
from einops import rearrange
import numpy as np
from monai.networks.layers.utils import get_act_layer, get_norm_layer
import torch.nn.functional as F

class PatchEmbeddingBlock(nn.Module):
    def __init__(self, in_channels: int,

                 patch_size: int,
                 emb_dim: int,
                 num_heads: int,
                 dropout_rate: float = 0.0,
                 spatial_dims: int = 3,
                 ) -> None:
        super().__init__()

        # num_patches = (image_size // patch_size) ** 3
        # patch_dim = in_channels * patch_size ** 3
        # self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches, emb_dim))
        self.dropout = nn.Dropout(dropout_rate)
        self.patch_embeddings = nn.Sequential(
            nn.Conv3d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(start_dim=2)
        )
        # trunc_normal_(self.position_embeddings, mean=0.0, std=0.02, a=-2.0, b=2.0)
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
        # embeddings = x + self.position_embeddings
        embeddings = self.dropout(x)
        return embeddings


class ViT(nn.Module):
    def __init__(self, patch_size=4,  in_channels=1, emb_dim=64, depth=4, heads=8,
                 mlp_dim=256, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embedding_block = PatchEmbeddingBlock(in_channels=in_channels,
                                                         patch_size=patch_size, emb_dim=emb_dim, num_heads=heads,
                                                         dropout_rate=dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size=emb_dim, mlp_dim=mlp_dim, num_heads=heads, dropout_rate=dropout
                              ) for i in range(depth)]
        )
        self.norm = nn.LayerNorm(emb_dim)

        # self.decode_blocks = nn.ModuleList(
        #     [nn.Sequential(
        #         get_norm_layer('instance', spatial_dims=3, channels=emb_dim // 2 ** i),
        #         get_act_layer('prelu'),
        #         nn.ConvTranspose3d(in_channels=emb_dim // 2 ** i,
        #                            out_channels=emb_dim // 2 ** (i + 1), kernel_size=2,
        #                            stride=2)
        #     ) for i in range(patch_size // 2)],
        # )
        self.final_conv = nn.Sequential(
            get_norm_layer('instance', spatial_dims=3,
                           channels=emb_dim),
            get_act_layer('prelu'),
            nn.Conv3d(emb_dim , 1 ,
                      kernel_size=1)
        )

    def forward(self, x):
        b,c,d,h,w = x.shape
        x= self.patch_embedding_block(x)
        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)
        # decoder
        x = x.transpose(-1,-2)
        x = x.reshape(shape=(x.shape[0],-1, d // self.patch_size, h // self.patch_size , w // self.patch_size))
        x = self.final_conv(x)
        # X: [B Patch C]
        # return x, hidden_states_out
        output = F.interpolate(x, scale_factor=self.patch_size)
        return output
