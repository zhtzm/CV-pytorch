from torch import Tensor
import torch
from torch import nn


def drop_path(X: Tensor, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:  # drop_prob=0代表不需要drop,not training表示在预测不能drop
        return X
    
    keep_prob = 1- drop_prob
    shape = (X.shape[0],) + (1,) * (X.ndim - 1)
    """
    这里若输入X(b, c, h, w)
    输出shape为(b, 1, 1, 1)
    (X.shapw[0],) == (b,)
    (1,) * (X.ndim - 1) == (1, 1, 1)
    相加为(b, 1, 1, 1)
    这是为了后面发挥广播机制
    """
    random_tensor = keep_prob + torch.rand(shape, dtype=X.dtype, device=X.device)
    random_tensor.floor_()
    """
    torch.rand(shape, dtype=X.dtype, device=X.device)按均匀分布生成tensor
    上面不妨假设b为4
    这里生成(0., 0.3333, 0.6667, 1.)实际生成不会这么规整，这里这样方便解释
    假设keep_prob为0.75
    random_tensor = (0.75, 1.0833, 1.4133, 1.75)
    向下取整得到(0., 1., 1., 1.)
    若这里对应X中的4个batch,显然我们保留了三个,实现了drop_prob=0.25
    """
    output = X.div(keep_prob) * random_tensor
    """
    这里X.div(keep_prob)可以理解为我们想要保证X的整体值特性不变去掉drop_prob
    X=(x1, x2, x3, x4) random_tensor=(0, 1, 1, 1)
    X.div(keep_prob)=(4x1/3, 4x2/3, 4x3/3, 4x4/3)
    output=(0, 4x2/3, 4x3/3, 4x4/3)
    """
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        assert self.drop_prob is not None
        return drop_path(x, self.drop_prob, self.training)
    