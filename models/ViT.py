from monai.networks.blocks.transformerblock import TransformerBlock
from monai.networks.layers import Conv, trunc_normal_
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