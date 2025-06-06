import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Front-End Convolution Module
class FrontEndConv(nn.Module):
    def __init__(self, in_nc: int, c: int, w: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(in_nc, c, kernel_size=w, stride=1, padding=(w-1)//2)
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='leaky_relu')
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_nc, T]
        # Returns: [B, c, T]
        return self.conv(x)

# Gated Residual Block (GRB) used in Speech Encoder
class GRB(nn.Module):
    def __init__(self, c: int, dilation: int = 1):
        super().__init__()
        self.conv_dilated = nn.Conv1d(c, c, kernel_size=3, stride=1,
                                      dilation=dilation, padding=dilation)
        self.res_conv = nn.Conv1d(c, c, kernel_size=1, stride=1, padding=0)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        nn.init.kaiming_normal_(self.conv_dilated.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.res_conv.weight, nonlinearity='leaky_relu')
        if self.conv_dilated.bias is not None:
            nn.init.zeros_(self.conv_dilated.bias)
        if self.res_conv.bias is not None:
            nn.init.zeros_(self.res_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, c, T]
        out = self.lrelu(self.conv_dilated(x))
        res = self.res_conv(x)
        return out + res

# Speech Encoder: stack of multiple GRB blocks
class SpeechEncoder(nn.Module):
    def __init__(self, c: int, Ns: int = 3):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(Ns):
            dilation = 2 ** i
            self.blocks.append(GRB(c, dilation))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, c, T]
        for block in self.blocks:
            x = block(x)
        return x  # [B, c, T]

# Residual Dense Block (RDB) used in FG Agent
class ResidualDenseBlock(nn.Module):
    def __init__(self, c: int, gc: int):
        super().__init__()
        self.conv1 = nn.Conv1d(c, gc, kernel_size=3, stride=1, dilation=1, padding=1)
        self.conv2 = nn.Conv1d(c + gc, gc, kernel_size=3, stride=1, dilation=2, padding=2)
        self.conv3 = nn.Conv1d(c + 2*gc, gc, kernel_size=3, stride=1, dilation=4, padding=4)
        self.conv4 = nn.Conv1d(c + 3*gc, gc, kernel_size=3, stride=1, dilation=8, padding=8)
        self.conv5 = nn.Conv1d(c + 4*gc, c, kernel_size=1, stride=1, padding=0)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv3.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv4.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv5.weight, nonlinearity='leaky_relu')
        for conv in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]:
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, c, T]
        x1 = self.lrelu(self.conv1(x))                         # [B, gc, T]
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), dim=1))) # [B, gc, T]
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), dim=1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), dim=1))
        return x5 + x  # Residual connection

# FG Agent: stack of RDBs + LayerNorm + LReLU, then predict mu and logvar
class FGAgent(nn.Module):
    def __init__(self, c: int, gc: int, Nf: int, Nr: int, m: int):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(Nf):
            self.blocks.append(ResidualDenseBlock(c, gc))
            self.blocks.append(nn.LayerNorm([c, 1]))
            self.blocks.append(nn.LeakyReLU(0.2, inplace=True))
        self.proj = nn.Conv1d(c, 2 * c * m, kernel_size=1)
        nn.init.kaiming_normal_(self.proj.weight, nonlinearity='leaky_relu')
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
        self.c = c
        self.m = m
        self.gauss_norm_const = 1 / np.sqrt(2 * np.pi)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, c, T]
        for layer in self.blocks:
            if isinstance(layer, nn.LayerNorm):
                x_mean = torch.mean(x, dim=2, keepdim=True)  # [B, c, 1]
                x = layer(x_mean)
                x = x.repeat(1, 1, x_mean.size(2))
            else:
                x = layer(x)
        x_pool = torch.mean(x, dim=2, keepdim=True)  # [B, c, 1]
        proj_out = self.proj(x_pool)  # [B, 2*c*m, 1]
        proj_out = proj_out.view(x.size(0), 2, self.c * self.m)  # [B, 2, c*m]
        mu = proj_out[:, 0, :]       # [B, c*m]
        logvar = proj_out[:, 1, :]   # [B, c*m]
        std = torch.exp(logvar * 0.5)
        eps = torch.randn_like(std)
        action_flat = mu + std * eps  # [B, c*m]
        action = action_flat.view(x.size(0), self.c, self.m)  # [B, c, m]
        logvar_clamped = torch.clamp(logvar, min=-10.0, max=10.0)
        log_prob = (
            torch.log(torch.tensor(self.gauss_norm_const, device=mu.device))
            - 0.5 * logvar_clamped
            - 0.5 * (eps ** 2)
        )  # [B, c*m]
        log_prob_sum = torch.sum(log_prob, dim=1)  # [B]
        return mu.view(x.size(0), self.c, self.m), std.view(x.size(0), self.c, self.m), action, log_prob_sum

# Dynamic Convolution Model combining FE, SE, FGAgent
class DynamicFilterModel(nn.Module):
    def __init__(self,
                 in_nc: int = 1,
                 c: int = 64,
                 w: int = 3,
                 Ns: int = 3,
                 gc: int = 32,
                 Nf: int = 4,
                 Nr: int = 4,
                 m: int = 3):
        super().__init__()
        self.fe = FrontEndConv(in_nc, c, w)
        self.se = SpeechEncoder(c, Ns)
        self.fg = FGAgent(c, gc, Nf, Nr, m)
        self.padding = (m - 1) // 2
        self.c = c
        self.m = m

    def forward(self, x: torch.Tensor, deterministic: bool=False) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        # x: [B, 1, T]
        fea = self.fe(x)               # [B, c, T]
        enc = self.se(fea)            # [B, c, T]
        mu, std, action, log_prob = self.fg(enc)  # shapes: [B, c, m], [B, c, m], [B, c, m], [B]
        if deterministic:
            action = mu
        outputs = []
        for i in range(x.size(0)):
            inp = enc[i].unsqueeze(0)              # [1, c, T]
            ker = action[i].unsqueeze(1)           # [c, 1, m]
            conv_out = F.conv1d(inp, ker, padding=self.padding, groups=self.c)  # [1, c, T]
            summed = torch.sum(conv_out, dim=1, keepdim=True)  # [1, 1, T]
            outputs.append(summed)
        outputs = torch.cat(outputs, dim=0)  # [B, 1, T]
        return (mu, std), outputs, log_prob
