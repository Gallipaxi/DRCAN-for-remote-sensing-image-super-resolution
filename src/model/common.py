import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res = res + x
        return res

# WDSR部分
class WDSR(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, wn, act=nn.ReLU(True), res_scale=1):
        super(WDSR, self).__init__()
        self.res_scale = res_scale
        body = []
        expand = 6
        linear = 0.8
        body.append(
            wn(conv(n_feats, n_feats * expand, kernel_size=1)))
        body.append(act)
        body.append(
            wn(conv(n_feats * expand, int(n_feats * linear), kernel_size=1)))
        body.append(
            wn(conv(int(n_feats * linear), n_feats, kernel_size=kernel_size)))

        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res = res + x
        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


# CBAM
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        #self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        #self.bn2 = nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.ca(out) * out


        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        return out


class Head_Block(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Head_Block, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        out = x + out
        new_residual = out
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.ca(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)


        out = out + new_residual
        return out

class WDSR_ATT(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, wn, act=nn.ReLU(True), res_scale=1, stride=1, downsample=None):
        super(WDSR_ATT, self).__init__()
        self.WDSR = WDSR(conv, n_feats, kernel_size, wn, act, res_scale)
        self.ATT = BasicBlock(n_feats, n_feats, stride, downsample)

    def forward(self, x):
        out = self.WDSR(x)
        out = self.ATT(out)

        return out

class LR_conv(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, wn, act, res_scale):
        super(LR_conv, self).__init__()
        self.WDSR = WDSR(conv, n_feats, kernel_size, wn, act, res_scale)
        self.conv = conv3x3(n_feats, n_feats)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.WDSR(x)
        out = self.conv(residual)
        out = self.relu(out)
        out = self.conv(out)
        # out = out + residual
        return out, residual

class HR_conv(nn.Module):
    def __init__(self, n_feats):
        super(HR_conv, self).__init__()
        self.conv = conv3x3(n_feats, n_feats)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = self.conv(out)

        # 增加一层
        residual = x + out
        out = self.conv(residual)
        out = self.relu(out)
        out = self.conv(out)
        return out, residual

class LR_block(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, wn, act, res_scale):
        super(LR_block, self).__init__()
        self.conv_block = LR_conv(conv, n_feats, kernel_size, wn, act, res_scale)
        self.ca = ChannelAttention(n_feats)
        self.conv = conv3x3(n_feats, n_feats)

        length = 2
        self.weight_1 = nn.Parameter(torch.ones(length))
        self.weight_2 = nn.Parameter(torch.ones(length))

    def forward(self, x):
        weight_1 = F.softmax(self.weight_1, 0)
        weight_2 = F.softmax(self.weight_2, 0)

        x_1, residual_1 = self.conv_block(x)
        w_1 = self.ca(x_1)
        x_1 = w_1 * x_1
        x_1 = x_1 + residual_1

        x_2, residual_2 = self.conv_block(x_1)
        w_2 = weight_1[0] * w_1 + weight_1[1] * self.ca(x_2)
        x_2 = w_2 * x_2
        x_2 = x_2 + residual_2

        x_3, residual_3 = self.conv_block(x_2)
        w_3 = weight_2[0] * w_2 + weight_2[1] * self.ca(x_3)
        x_3 = w_3 * x_3
        x_3 = x_3 + residual_3

        return x_3

class HR_block(nn.Module):
    def __init__(self, n_feats):
        super(HR_block, self).__init__()
        self.conv_block = HR_conv(n_feats)
        self.ca = ChannelAttention(n_feats)
        self.conv = conv3x3(n_feats, n_feats)
        self.relu = nn.ReLU(inplace=True)

        length = 2
        self.weight_1 = nn.Parameter(torch.ones(length))
        self.weight_2 = nn.Parameter(torch.ones(length))

    def forward(self, x):
        weight_1 = F.softmax(self.weight_1, 0)
        weight_2 = F.softmax(self.weight_2, 0)

        x_1, residual_1 = self.conv_block(x)
        w_1 = self.ca(x_1)
        x_1 = w_1 * x_1
        x_1 = x_1 + residual_1

        x_2, residual_2 = self.conv_block(x_1)
        w_2 = weight_1[0] * w_1 + weight_1[1] * self.ca(x_2)
        x_2 = w_2 * x_2
        x_2 = x_2 + residual_2

        x_3, residual_3 = self.conv_block(x_2)
        w_3 = weight_2[0] * w_2 + weight_2[1] * self.ca(x_3)
        x_3 = w_3 * x_3
        x_3 = x_3 + residual_3

        return x_3