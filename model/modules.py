#!/usr/bin/env python
# coding=utf-8
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# from model.sinclayer import SincConv
EPS = 1e-8


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
            if m.bias is not None:
                m.bias.data.zero_()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-8):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma.view(1, -1, 1) * (x - mean) / (std + self.eps) + self.beta.view(1, -1, 1)


class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32, no=64, normalize=1):
        super(ResidualDenseBlock, self).__init__()
        self.normalize = normalize
        self.conv1 = nn.Conv1d(nf, gc, kernel_size=3, stride=1, dilation=1, padding=1)
        self.conv2 = nn.Conv1d(nf + gc, gc, kernel_size=3, stride=1, dilation=2, padding=2)
        self.conv3 = nn.Conv1d(nf + 2 * gc, gc, kernel_size=3, stride=1, dilation=4, padding=4)
        self.conv4 = nn.Conv1d(nf + 3 * gc, gc, kernel_size=3, stride=1, dilation=8, padding=8)
        self.conv5 = nn.Conv1d(nf + 4 * gc, nf, kernel_size=1, stride=1, dilation=1, padding=0)
        self.conv6 = nn.Conv1d(nf, no, kernel_size=16, stride=8, dilation=1, padding=4)
        if normalize:
            self.normal = LayerNorm((no, 1))
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.shortcut = nn.Conv1d(nf, nf, kernel_size=1, stride=1, dilation=1, padding=0)

        initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.shortcut], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1)) + self.shortcut(x)
        if self.normalize:
            #            x6 = self.conv6(x5)
            #            x6 = self.normal(x6)
            x6 = self.lrelu(self.normal(self.conv6(x5)))
        else:
            x6 = self.lrelu(self.conv6(x5))
        return x6


class Base_model(nn.Module):
    def __init__(self, in_nc, out_nc, nf, gc, times, normalize):
        super(Base_model, self).__init__()
        self.times = times

        self.conv_first = nn.Conv1d(in_nc, nf, kernel_size=3, stride=1, dilation=1, padding=1)

        for t in range(self.times):
            if t == self.times - 2:
                setattr(self, f'b_{t}', ResidualDenseBlock(nf, gc, nf * 2, normalize))
            elif t == self.times - 1:
                setattr(self, f'b_{t}', ResidualDenseBlock(nf * 2, gc * 2, nf * 6, normalize))
            else:
                setattr(self, f'b_{t}', ResidualDenseBlock(nf, gc, nf, normalize))

        self.pooling = nn.AdaptiveAvgPool1d(1)

        # self.gauss_norm_const = torch.tensor((1.0 / np.sqrt(2 * math.pi)).astype(np.float32)).cuda()
        self.register_buffer("gauss_norm_const", torch.tensor(1 / np.sqrt(2 * np.pi), dtype=torch.float32))

        initialize_weights([self.conv_first], 1)

    def forward(self, x):
        fea = self.conv_first(x)
        feas = fea
        for t in range(self.times):
            feas = getattr(self, f'b_{t}')(feas)

        out = self.pooling(feas).view(fea.size(0), -1, 2)

        # device-agnostic 난수
        random_var = torch.randn_like(out[:, :, 0])
        mu = out[:, :, 0]
        logvar = out[:, :, 1]

        logvar_clamped = torch.clamp(logvar, min=-10.0, max=10.0)

        random_var = torch.randn((mu.size(0), mu.size(1)), device=mu.device)
        std = torch.exp(logvar_clamped / 2.0)

        actionn = mu + std * random_var  # [B, channel_gauss]
        action = actionn.view(fea.size(0), fea.size(1), -1)  # ([B, C_feat, K])

        action_prob = (
                torch.log(self.gauss_norm_const.to(mu.device))
                - 0.5 * logvar_clamped
                - 0.5 * torch.pow((actionn - mu), 2) / torch.exp(logvar_clamped)
        )  # [B, channel_gauss]

        #    fea[i]:  [C_feat, T]
        #    action[i]: [C_feat, K]
        x1 = []
        for i in range(out.size(0)):
            conv_input = fea[i].unsqueeze(0)  # [1, C_feat, T]
            conv_ker = action[i].unsqueeze(0)  # [1, C_feat, K]
            conv_out = F.conv1d(conv_input, conv_ker, padding=1)
            # conv_out: [1, 1, T] (channel_dim=1로 축소)
            x1.append(conv_out)

        outputs = torch.cat(x1, dim=0)  # [B, 1, T]  (또는 [B, C_out, T], 모델 설계에 따라 달라질 수 있음)

        return out, outputs, action_prob