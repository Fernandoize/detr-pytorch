# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
from typing import Dict, List

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter

from .ops import NestedTensor, is_main_process


class PositionEmbeddingSine(nn.Module):
    """
    这是一个更标准的位置嵌入版本，按照sine进行分布
    采用sin+cos表示位置向量
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        # 嵌入向量的维度
        self.num_pos_feats  = num_pos_feats
        # 温度参数，用于控制正弦函数的周期性
        self.temperature    = temperature
        # 是否对位置进行归一化
        self.normalize      = normalize
        # 如果没有传入scale，则使用默认的2π
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x           = tensor_list.tensors
        # 提取输入的掩码（用于表示图像中有效区域）
        mask        = tensor_list.mask
        assert mask is not None

        # # 获取mask的反向（有效区域）
        not_mask    = ~mask
        # 计算嵌入位置的累加值 沿着 y 轴（高度）累加
        y_embed     = not_mask.cumsum(1, dtype=torch.float32)
        # 沿着 x 轴（宽度）累加
        x_embed     = not_mask.cumsum(2, dtype=torch.float32)

        # 归一化位置嵌入（如果需要）
        if self.normalize:
            eps     = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        # 计算正弦和余弦函数的嵌入
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        #  计算正弦和余弦嵌入
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        # 合并 x 和 y 方向的嵌入，并调整形状
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

class PositionEmbeddingLearned(nn.Module):
    """
    创建可学习的位置向量，用于表示输入特征图中每个位置的位置信息
    """
    def __init__(self, num_pos_feats=256):
        """
        num_pos_feats=256：表示每个位置编码的维度，这里设定为 256 维
        row_embed = nn.Embedding(50, num_pos_feats)：创建一个可学习的嵌入层，用于表示图像中每一行的位置。
        50 是嵌入矩阵的大小，即最多支持 50 行的输入图像。
        col_embed = nn.Embedding(50, num_pos_feats)：创建一个可学习的嵌入层，用于表示图像中每一列的位置。
        50 是嵌入矩阵的大小，即最多支持 50 列的输入图像。
        self.reset_parameters()：初始化嵌入层的参数。接下来会在 reset_parameters 方法中进行初始化。
        """
        super().__init__()
        # embedding本质是一个查找表，其大小为(50, num_pos_feats)的矩阵，当我输入一个x = (3, 2)的词汇时，此
        # 时会根据向量索引在embedding中进行查找，输入大小则为(3, 2, num_pos_feats),意味着embedding的词汇中向量只能为 <= 50的整数

        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    # 输出为(b, d, h, w)
    def forward(self, tensor_list: NestedTensor):
        # x 是输入的图像张量，其形状是 (batch_size, channels, height, width)，即一个批次的图像
        x       = tensor_list.tensors
        # h, w = x.shape[-2:]：获取输入图像的高度 (h) 和宽度 (w)，这两个值将用于生成行列位置嵌入
        h, w    = x.shape[-2:]
        # 创建一个长度为 w 的张量 i，表示列的位置（i 为从 0 到 w-1 的整数）。这些列的位置将被用来生成列方向的嵌入。
        i       = torch.arange(w, device=x.device)
        # 创建一个长度为 h 的张量 j，表示行的位置（j 为从 0 到 h-1 的整数）。这些行的位置将被用来生成行方向的嵌入
        j       = torch.arange(h, device=x.device)
        # 将列位置 i 输入到 col_embed 嵌入层，得到列方向的位置嵌入（x_emb）。
        x_emb   = self.col_embed(i)
        # 将行位置 j 输入到 row_embed 嵌入层，得到行方向的位置嵌入（y_emb）。
        y_emb   = self.row_embed(j)
        # 将列和行的嵌入沿着最后一个维度（特征维度）拼接在一起，形成一个新的位置嵌入张量。拼接后的形状为 (height, width, 2 * num_pos_feats)。
        # x_emb (w, num_pos_feats) y (h, num_pos_feats)
        # unsqueeze + repeat后得到(h, w, num_pos_feats) 行列的位置向量对齐
        # concat后得到(h, w, num_pos_feats * 2)
        # unsqueeze 插入batch 维度. 并repeat保证每个样本都有位置向量
        pos     = torch.cat([
            # unsqueeze(0).repeat(h, 1, 1)：对列嵌入进行扩展，以便将列嵌入沿着行维度复制 h 次，从而使每一行的所有列共享相同的列位置嵌入。
            # 插入一个新的维度，用来调整张量的形状,例如(2, 2).unsqueeze(0) 变成(1, 2, 2) 再repeat(2, 1, 1)后变成(2, 2, 2)
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            # unsqueeze(1).repeat(1, w, 1)：对行嵌入进行扩展，以便将行嵌入沿着列维度复制 w 次，从而使每一列的所有行共享相同的行位置嵌入
            y_emb.unsqueeze(1).repeat(1, w, 1),
            # pos = ... .permute(2, 0, 1)：调整位置嵌入的维度顺序，将位置嵌入转换为 (2 * num_pos_feats, h, w) 的形状。
            # unsqueeze(0).repeat(x.shape[0], 1, 1, 1)：将位置嵌入扩展到整个批次（batch size）。最终的输出 pos 形状为 (batch_size, 2 * num_pos_feats, h, w)，它包含了整个批次中每个像素的位置嵌入。
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos

def build_position_encoding(position_embedding, hidden_dim=256):
    # 创建位置向量
    N_steps = hidden_dim // 2
    if position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {position_embedding}")

    return position_embedding

class FrozenBatchNorm2d(torch.nn.Module):
    """
    冻结固定的BatchNorm2d。
    """
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        w       = self.weight.reshape(1, -1, 1, 1)
        b       = self.bias.reshape(1, -1, 1, 1)
        rv      = self.running_var.reshape(1, -1, 1, 1)
        rm      = self.running_mean.reshape(1, -1, 1, 1)
        eps     = 1e-5
        scale   = w * (rv + eps).rsqrt()
        bias    = b - rm * scale
        return x * scale + bias

class BackboneBase(nn.Module):
    """
    用于指定返回哪个层的输出
    这里返回的是最后一层
    """
    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers   = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers   = {'layer4': "0"}
            
        # 用于指定返回的层
        self.body           = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels   = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs                           = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m           = tensor_list.mask
            assert m is not None
            mask        = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name]   = NestedTensor(x, mask)
        return out

class Backbone(BackboneBase):
    """
    ResNet backbone with frozen BatchNorm.
    """
    def __init__(self, name: str, train_backbone: bool, return_interm_layers: bool, dilation: bool, pretrained:bool):
        # 首先利用torchvision里面的model创建一个backbone模型
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation    = [False, False, dilation],
            pretrained                      = pretrained, 
            norm_layer                      = FrozenBatchNorm2d
        )
        # 根据选择的模型，获得通道数
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)

class Joiner(nn.Sequential):
    """
    用于将主干和位置编码模块进行结合
    """
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs                      = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos                     = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos

def build_backbone(backbone, position_embedding, hidden_dim, train_backbone=True, pretrained=False):
    # 创建可学习的位置向量还是固定的按'sine'排布的位置向量
    position_embedding  = build_position_encoding(position_embedding, hidden_dim)
    # 创建主干
    backbone            = Backbone(backbone, train_backbone, False, False, pretrained=pretrained)
    
    # 用于将主干和位置编码模块进行结合
    model               = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model