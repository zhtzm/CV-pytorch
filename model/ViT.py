from collections import OrderedDict
from functools import partial
from torch import Tensor
import torch
import torch.nn as nn

from model.DropPath import DropPath


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 in_c=3,
                 embed_dim=768,
                 norm_layer=None):
        super(PatchEmbed, self).__init__()
        image_size = (image_size, image_size)
        patch_size = (patch_size, patch_size)
    
        self.img_size = image_size
        self.patch_size = patch_size
        # 按patch裁剪后的HxW
        self.grid_size = (self.img_size[0] // self.patch_size[0],
                          self.img_size[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, 
                              stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, X: Tensor):
        _, _, H, W = X.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        X = self.proj(X)
        X = X.flatten(2).transpose(1, 2)
        # 这里flatten展平第二维之后即H'维和W'维,此时shape=(B, C, H'W')=(B, C, num_patches)
        # transpose用于交换轴,这里交换1,2轴,shape变为(B, num_patches, C),可理解为每个batch中做转置
        X = self.norm(X)

        return X
    
class Attention(nn.Module):
    def __init__(self,
                 dim,   # embed_dim,也指下面的C
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5   
        # attention = softmax(QTV/sqrt(d)) @ V, 这个scale就是指代sqrt(d)这个用于调节的参数,当然我们可以给定超参数
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # 这里应该时直接通过一个升维,我们直接从中截取出qkv,这里就用到XW等效cat(QW1, KW2, VW3)的原理
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, X):
        B, N, C = X.shape
        # 一般理解这里的X是embedding(patch_embedding+position_embedding)

        qkv = self.qkv(X).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # qkv(): -> [B, N, 3 * C]
        # reshape: -> [B, N, 3, num_heads, C // num_heads]
        # permute: -> [3, B, num_heads, N, C // num_heads]
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # shape(B, num_heads, N, N)
        X = (attn @ v).transpose(1, 2).reshape(B, N, C)
        X = self.proj(X)
        X = self.proj_drop(X)
        # shape(B, N, C)

        return X
    

class MLP(nn.Module):
    def __init__(self, 
                 in_features, 
                 hidden_features=None, 
                 out_features=None, 
                 act_layer=nn.GELU, 
                 drop=0.):
        super(MLP, self).__init__()
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 mlp_ratio=4.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()   
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim=dim,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio,
                              proj_drop_ratio=drop_ratio)
        self.drop_path = DropPath(drop_prob=drop_path_ratio) if drop_path_ratio > 0. \
            else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop_ratio)
        
    def forward(self, X):
        X = X + self.drop_path(self.attn(self.norm1(X)))
        X = X + self.drop_path(self.mlp(self.norm2(X)))
        return X


class VisionTransformer(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 distilled=False,
                 embed_dim=768,
                 drop_ratio=0.,
                 act_layer=None,
                 norm_layer=None,
                 image_size=224,
                 patch_size=16,
                 in_c=3,
                 embed_layer=PatchEmbed,
                 num_heads=12,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_path_ratio=0., 
                 depth=12,
                 attn_drop_ratio=0.,
                 representation_size=None, 
                 ):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(image_size=image_size,
                                       patch_size=patch_size,
                                       in_c=in_c,
                                       embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, 
                  num_heads=num_heads, 
                  mlp_ratio=mlp_ratio, 
                  qkv_bias=qkv_bias, 
                  qk_scale=qk_scale,
                  drop_ratio=drop_ratio, 
                  attn_drop_ratio=attn_drop_ratio, 
                  drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, 
                  act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x
    

def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
