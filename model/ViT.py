from copy import deepcopy
import numpy as np
from torch import Tensor
import torch
import torch.nn as nn
import torch.functional as F


class VisionTransformer(nn.Module):
    def __init__(self,
                 image_size: int,
                 patch_size: int,
                 num_layers: int,
                 num_heads: int,
                 in_channel:int,
                 hidden_dim: int,
                 mlp_dim: int,
                 dropout: float = 0.0,
                 attention_dropout: float = 0.0,
                 num_classes: int = 1000
                ):
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbed(image_size, patch_size, in_channel, hidden_dim)
        seq_length = self.patch_embed.num_patches

        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1

        self.pos_enc = PositionalEncodering(seq_length, hidden_dim)
        mlp_layer = MLP(in_dim=hidden_dim, hidden_dim=mlp_dim, drop=dropout)
        d_k = d_v = hidden_dim // num_heads
        assert d_k * num_heads == hidden_dim
        encoder_layer = TransformerEncoderLayer(hidden_dim, num_heads, d_k, d_v, mlp_layer, attention_dropout, dropout, nn.Dropout, True)
        norm = nn.LayerNorm(hidden_dim)
        self.encoder = TransformerEncoder(encoder_layer, num_layers, dropout, norm)

        self.heads = nn.Linear(hidden_dim, num_classes)
        self.softmax = nn.Softmax(dim=-1)

        self._init_vit_weights()

    def _init_vit_weights(self):
        for m in self: 
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

    def forward(self, x):
        x = self.patch_embed(x)
        n = x.shape[0]
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = x[:, 0]
        x = self.heads(x)
        x = self.softmax(x)
        return x
    

class PositionalEncodering(nn.Module):
    def __init__(self, seq_length, hidden_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))

    def forward(self, x):
        """
            input: x(B, seq_length, E)
            output: x(B, seq_length, E)
        """
        return x + self.pos_embedding


class PatchEmbed(nn.Module):
    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 in_c=3,
                 embed_dim=768):
        super(PatchEmbed, self).__init__()
        image_size = (image_size, image_size)
        patch_size = (patch_size, patch_size)
    
        self.img_size = image_size
        self.patch_size = patch_size
        self.grid_size = (self.img_size[0] // self.patch_size[0],
                          self.img_size[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.conv_proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, 
                              stride=patch_size)

    def forward(self, X: Tensor):
        """
            input: X(B, C, H, W)
            output: X(B, N, embeding)
        """
        _, _, H, W = X.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        X = self.conv_proj(X)
        X = X.flatten(2).transpose(1, 2)

        return X
    

class MLP(nn.Module):
    def __init__(self, 
                 in_dim: int, 
                 hidden_dim: int, 
                 act_layer=nn.GELU, 
                 drop=0.):
        super(MLP, self).__init__()
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_dim, in_dim)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        """
            input: x(B, N, in_dim)
            output: x(B, N, in_dim)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self,
                 encoder_layer: nn.Module,
                 num_encoder_layers: int,
                 dropout: float = 0.0,
                 norm: nn.Module = None
                 ) -> None:
        super(TransformerEncoder, self).__init__()
        self.drop = nn.Dropout(p=dropout)
        self.layers = _get_clones(encoder_layer, num_encoder_layers)
        self.norm = norm

    def forward(self, enc_inputs):  
        enc_outputs = self.drop(enc_inputs)       
        for layer in self.layers:
            enc_outputs = layer(enc_outputs)  
        if self.norm is not None:
            enc_outputs = self.norm(enc_outputs)
        return enc_outputs


class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_head: int, 
                 d_k: int, 
                 d_v: int,
                 mlp_layer: nn.Module,
                 attention_dropout: float = 0.,
                 dropout: float = 0.,
                 drop_layer: nn.Module = nn.Dropout,
                 norm_first: bool = False
                 ) -> None:
        super(TransformerEncoderLayer, self).__init__()
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, n_head, d_k, d_v, attention_dropout)
        self.dropout = drop_layer(p=dropout) if dropout > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = mlp_layer

    def forward(self, enc_inputs):
        """
            inputs:
                enc_inputs: (batch_size, src_len, d_model)
            outputs:
                enc_outputs: (batch_size, src_len, d_model)
                attn: (batch_size, n_heads, src_len, src_len)
        """
        if self.norm_first:
            residual = enc_inputs
            enc_outputs = self.norm1(enc_inputs)
            enc_outputs, _ = self.attention(enc_outputs, enc_outputs, enc_outputs)
            enc_outputs = self.dropout(enc_outputs)
            enc_outputs += residual

            residual = enc_outputs
            enc_outputs = self.norm2(enc_outputs)
            enc_outputs = self.mlp(enc_outputs)
            enc_outputs += residual
        else:
            residual = enc_inputs
            enc_outputs, _ = self.attention(enc_inputs, enc_inputs, enc_inputs)
            enc_outputs = self.dropout(enc_outputs)
            enc_outputs += residual
            enc_outputs = self.norm1(enc_outputs)

            residual = enc_outputs
            enc_outputs = self.mlp(enc_outputs)
            enc_outputs += residual
            enc_outputs = self.norm2(enc_outputs)
        
        return enc_outputs


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_head: int,
                 d_k: int,
                 d_v: int,
                 p_dropout: float = 0.
                 ):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.W_Q = nn.Linear(d_model, d_k * n_head, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_head, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_head, bias=False)
        self.sdpa = ScaledDotProductAttention(p_dropout)
        self.fc = nn.Linear(d_v * n_head, d_model, bias=False)

    def forward(self, Q: Tensor, K: Tensor, V: Tensor):
        """
            inputs:
                Q: (batch_size, seq_len1, d_model)
                K: (batch_size, seq_len2, d_model)
                V: (batch_size, seq_len2, d_model]
                mask: (batch_size, seq_len1, seq_len2)
            outputs:
                output: (batch_size, seq_len1, d_model)
                attn: (batch_size, n_heads, seq_len1, seq_len2)
        """
        batch_size = Q.shape[0]
        # Q -> (batch_size, seq_len1, d_k * n_head) -> (batch_size, seq_len1, n_head, d_k) -> (batch_size, n_head, seq_len1, d_k)
        Q = self.W_Q(Q).view(batch_size, -1, self.n_head, self.d_k).transpose(1,2)
        # K -> (batch_size, seq_len2, d_k * n_head) -> (batch_size, seq_len2, n_head, d_k) -> (batch_size, n_head, seq_len2, d_k)  
        K = self.W_K(K).view(batch_size, -1, self.n_head, self.d_k).transpose(1,2)
        # V -> (batch_size, seq_len2, d_v * n_head) -> (batch_size, seq_len2, n_head, d_v) -> (batch_size, n_head, seq_len2, d_v)
        V = self.W_V(V).view(batch_size, -1, self.n_head, self.d_v).transpose(1,2)
        # mask -> (batch_size, 1, seq_len1, seq_len2) -> (batch_size, n_heads, seq_len1, seq_len2)                  
        context, attn = self.sdpa(Q, K, V)             

        # (batch_size, seq_len1, n_heads * d_v)                                                                      
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_head * self.d_v)      
        output = self.fc(context)    

        return output, attn
    

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout) -> None:
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(-1)

    def forward(self, Q: Tensor, K: Tensor, V: Tensor):
        """
            inputs:
                Q: (batch_size, n_head, seq_len1, d_k)
                K: (batch_size, n_head, seq_len2, d_k)  
                V: (batch_size, n_head, seq_len2, d_v)
                mask: (batch_size, n_heads, seq_len1, seq_len2)
            outputs:
                context: (batch_size, n_heads, seq_len1, d_v)
                attn: (batch_size, n_heads, seq_len1, seq_len2)
        """
        # scores (batch_size, n_heads, seq_len1, seq_len2)
        scores = torch.matmul(Q, K.permute(0, 1, 3, 2)) / np.sqrt(Q.shape[-1])
        # attn (batch_size, n_heads, seq_len1, seq_len2)
        attn = self.softmax(scores)
        self.dropout(attn)
        # context (batch_size, n_heads, seq_len1, d_v)
        context = torch.matmul(attn, V)                                 
        return context, attn
    

def _get_clones(module, N):
    return nn.ModuleList([deepcopy(module) for i in range(N)])


def vit_b_16(image_size, num_classes):
    return VisionTransformer(
        image_size=image_size,
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        dropout=0.1,
        attention_dropout=0.1,
        num_classes=num_classes
    )


def vit_b_32(image_size, num_classes):
    return VisionTransformer(
        image_size=image_size,
        patch_size=32,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        dropout=0.1,
        attention_dropout=0.1,
        num_classes=num_classes
    )


def vit_l_16(image_size, num_classes):
    return VisionTransformer(
        image_size=image_size,
        patch_size=16,
        num_layers=24,
        num_heads=16,
        hidden_dim=1024,
        mlp_dim=4096,
        dropout=0.2,
        attention_dropout=0.2,
        num_classes=num_classes
    )


def vit_l_32(image_size, num_classes):
    return VisionTransformer(
        image_size=image_size,
        patch_size=32,
        num_layers=24,
        num_heads=16,
        hidden_dim=1024,
        mlp_dim=4096,
        dropout=0.2,
        attention_dropout=0.2,
        num_classes=num_classes
    )


def vit_h_14(image_size, num_classes):
    return VisionTransformer(
        image_size=image_size,
        patch_size=14,
        num_layers=32,
        num_heads=16,
        hidden_dim=1280,
        mlp_dim=5120,
        dropout=0.3,
        attention_dropout=0.3,
        num_classes=num_classes
    )