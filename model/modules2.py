import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

################################################################################
# Front-End → SE → FG Agent → Dynamic Filter 통합 모델
################################################################################

class FrontEndConv(nn.Module):
    """
    Front-End Feature Extraction (FE)
    - in_nc: 입력 채널(보통 1)
    - out_nc: FE 출력 채널(c)
    - kernel_size: 컨볼루션 커널 폭(w)
    """
    def __init__(self, in_nc: int, out_nc: int, kernel_size: int):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_nc, out_nc,
                              kernel_size=kernel_size,
                              stride=1,
                              padding=padding)
        self.act = nn.PReLU()
        # 가중치 초기화
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='leaky_relu')
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_nc, T]
        out = self.conv(x)    # [B, c, T]
        out = self.act(out)
        return out            # [B, c, T]


class GRB(nn.Module):
    """
    Gated Residual Block (GRB) for SE
    - channels: 입력·출력 채널 수(c)
    - dilation: dilated conv에서 사용할 dilation
    """
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        # 3×1 dilated conv
        self.conv_dilated = nn.Conv1d(channels, channels,
                                      kernel_size=3,
                                      stride=1,
                                      padding=dilation,
                                      dilation=dilation)
        # 1×1 residual conv
        self.conv_res = nn.Conv1d(channels, channels,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)
        self.act = nn.PReLU()
        # 초기화
        nn.init.kaiming_normal_(self.conv_dilated.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv_res.weight, nonlinearity='leaky_relu')
        if self.conv_dilated.bias is not None:
            nn.init.constant_(self.conv_dilated.bias, 0.0)
        if self.conv_res.bias is not None:
            nn.init.constant_(self.conv_res.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, c, T]
        out = self.conv_dilated(x)  # [B, c, T]
        out = self.act(out)
        res = self.conv_res(x)      # [B, c, T]
        out = out + res             # residual 연결
        out = self.act(out)
        return out                   # [B, c, T]


class SpeechEncoder(nn.Module):
    """
    Speech Encoder (SE)
    - n_blocks: GRB 블록을 몇 개 쌓을지 (Ns)
    - channels: GRB 내부 채널 수(c)
    """
    def __init__(self, channels: int, n_blocks: int):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            dilation = 2 ** i
            self.blocks.append(GRB(channels, dilation))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, c, T]
        out = x
        for block in self.blocks:
            out = block(out)       # [B, c, T] (시퀀스 길이 유지)
        return out                 # [B, c, T]


class ResidualDenseBlock(nn.Module):
    """
    Residual Dense Block (RDB) for FG Agent
    - channels: 입력 채널 수 (c)
    - growth: 내부 growth 채널 (gc)
    - n_dilated: dilated conv 레이어 개수 (nr)
    - out_channels: RDB 출력 채널 (여기서는 channels로 고정)
    """
    def __init__(self, channels: int, growth: int, n_dilated: int):
        super().__init__()
        self.n_dilated = n_dilated
        self.growth = growth
        in_ch = channels
        self.layers = nn.ModuleList()
        for i in range(n_dilated):
            d = 2 ** i
            self.layers.append(
                nn.Conv1d(in_ch, growth,
                          kernel_size=3,
                          stride=1,
                          padding=d,
                          dilation=d)
            )
            in_ch += growth
        # 최종 1×1 conv: 채널을 다시 channels로 축소
        self.conv_last = nn.Conv1d(in_ch, channels, kernel_size=1, stride=1, padding=0)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        # shortcut: 입력 채널 = channels → 출력 채널 = channels
        self.shortcut = nn.Conv1d(channels, channels, kernel_size=1, stride=1, padding=0)
        # 초기화
        for m in self.layers:
            nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        nn.init.kaiming_normal_(self.conv_last.weight, nonlinearity='leaky_relu')
        if self.conv_last.bias is not None:
            nn.init.constant_(self.conv_last.bias, 0.0)
        nn.init.kaiming_normal_(self.shortcut.weight, nonlinearity='leaky_relu')
        if self.shortcut.bias is not None:
            nn.init.constant_(self.shortcut.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, channels, T]
        cat_feats = x
        for layer in self.layers:
            out = layer(cat_feats)      # [B, growth, T]
            out = self.lrelu(out)
            cat_feats = torch.cat([cat_feats, out], dim=1)  # 채널 축 적층
        out = self.conv_last(cat_feats) + self.shortcut(x)  # [B, channels, T]
        out = self.lrelu(out)
        return out                   # [B, channels, T] (시퀀스 길이 유지)


class FGAgent(nn.Module):
    """
    Filter Generation Agent (Policy 네트워크)
    - channels: SE 출력 채널 c
    - kernel_size: 동적 컨볼루션 커널 크기 (m)
    - n_rdb: FG agent 내부 RDB 블록 개수 (Nf)
    - growth: RDB에서 사용할 growth 채널 (gc)
    - n_dilated: 각 RDB 내부 dilated conv 개수 (nr)
    """
    def __init__(self,
                 channels: int,
                 kernel_size: int,
                 n_rdb: int,
                 growth: int,
                 n_dilated: int):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.n_rdb = n_rdb

        # Nf개의 RDB 블록 쌓기 → 각각 [B, c, T] 형태 유지
        self.rdb_blocks = nn.ModuleList()
        for _ in range(n_rdb):
            self.rdb_blocks.append(ResidualDenseBlock(channels, growth, n_dilated))

        # 최종 μ, logvar 예측용 1×1 conv (입력 채널 = channels, 출력 채널 = 2*c*m)
        self.proj = nn.Conv1d(channels, 2 * channels * kernel_size,
                              kernel_size=1, stride=1, padding=0)
        nn.init.kaiming_normal_(self.proj.weight, nonlinearity='leaky_relu')
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0.0)

    def forward(self, x: torch.Tensor):
        # x: [B, c, T]
        out = x
        for block in self.rdb_blocks:
            out = block(out)  # 각 RDB 블록마다 [B, c, T]

        # 마지막 RDB 후에만 pooling → μ, logvar 예측
        pooled = F.adaptive_avg_pool1d(out, 1)  # [B, c, 1]
        stats = self.proj(pooled)             # [B, 2*c*m, 1]
        stats = stats.squeeze(-1)             # [B, 2*c*m]

        # μ, logvar 분리
        total_dim = self.channels * self.kernel_size
        mu     = stats[:, :total_dim]         # [B, c*m]
        logvar = stats[:, total_dim:]         # [B, c*m]

        # reparameterization
        logvar_clamped = torch.clamp(logvar, min=-10.0, max=10.0)
        std = torch.exp(0.5 * logvar_clamped)  # [B, c*m]
        eps = torch.randn_like(std)            # [B, c*m]
        action = mu + eps * std                # [B, c*m]

        # 독립 가우시안 로그확률 계산
        var = torch.exp(logvar_clamped)        # [B, c*m]
        const = -0.5 * np.log(2 * np.pi)
        log_prob_each = const - 0.5 * logvar_clamped - 0.5 * ((action - mu) ** 2) / var
        # [B, c*m]
        log_prob_sum = torch.sum(log_prob_each, dim=1)  # [B]

        # action을 [B, c, m]로 reshape
        action = action.view(-1, self.channels, self.kernel_size)  # [B, c, m]

        return mu, logvar_clamped, action, log_prob_sum


class DynamicFilterModel(nn.Module):
    """
    FE → SE → FGAgent → Dynamic Filtering 통합 모델
    - in_nc: 입력 채널 (1)
    - c: 내부 채널 수
    - w: FE 커널 크기
    - Ns: SE 블록 수
    - Nf: FG agent 블록 수
    - gc: FG agent 내부 growth 채널
    - nr: RDB 내부 dilated conv 수
    - m: 동적 필터 커널 크기
    """
    def __init__(self,
                 in_nc: int,
                 c: int,
                 w: int,
                 Ns: int,
                 Nf: int,
                 gc: int,
                 nr: int,
                 m: int):
        super().__init__()
        self.fe = FrontEndConv(in_nc, c, w)             # FE
        self.se = SpeechEncoder(c, Ns)                  # SE
        self.fg = FGAgent(c, m, Nf, gc, nr)             # FGAgent

    def forward(self, x: torch.Tensor, deterministic: bool = False):
        """
        x: [B, in_nc, T] (입력 noisy waveform)
        deterministic: True면 샘플링 대신 μ만 사용 (validation 모드)
        """
        fea = self.fe(x)         # [B, c, T]
        enc = self.se(fea)       # [B, c, T]

        mu, logvar, action, log_prob = self.fg(enc)
        # mu: [B, c*m], logvar: [B, c*m], action: [B, c, m], log_prob: [B]

        if deterministic:
            # validation 모드 → μ 사용 (action 대신)
            action = mu.view(-1, self.fg.channels, self.fg.kernel_size)

        # Dynamic convolution
        # for each sample in batch → groups=C로 conv 수행
        B, C, T = enc.size()
        k = self.fg.kernel_size
        pad = (k - 1) // 2

        enhanced_list = []
        for i in range(B):
            enc_i = enc[i].unsqueeze(0)          # [1, C, T]
            ker_i = action[i].unsqueeze(1)       # [C, 1, m]
            conv_i = F.conv1d(enc_i, ker_i,
                              padding=pad,
                              groups=C)          # [1, C, T]
            enh_i = torch.sum(conv_i, dim=1, keepdim=True)  # [1, 1, T]
            enhanced_list.append(enh_i)
        enhanced = torch.cat(enhanced_list, dim=0)  # [B, 1, T]

        # gauss 파라미터 묶기: [B, c*m, 2]
        gauss = torch.stack([mu, logvar], dim=2)    # [B, c*m, 2]

        # 반환: (gauss, enhanced, log_prob)
        return gauss, enhanced, log_prob
