import torch
from utils.metrics import pesq_wrapper as pesq

def Loss(gauss, data_predict, data_orig):
    # mu = gauss[:, :, 0]  # [B, channel_gauss]
    logvar = gauss[:, :, 1]  # [B, channel_gauss]

    logvar_clamped = torch.clamp(logvar, min=-20.0, max=20.0)  # overflow 방지
    var = torch.exp(logvar_clamped)  # [B, channel_gauss]
    # kl_loss = 0.5 * (mu ** 2 + var - 1 - logvar_clamped)

    n_l2 = torch.mean(torch.pow(data_predict - data_orig, 2), dim=[1, 2])
    s_l2 = torch.mean(torch.pow(data_orig, 2), dim=[1, 2])
    snr = 10 * torch.log10( (s_l2 + 1e-8) / (n_l2 + 1e-8) )

    avg_snr = torch.mean(snr)
    avg_mse = torch.mean(torch.sqrt(n_l2 + 1e-8))  # [B] → 평균
    # total_kl = torch.sum(kl_loss)  # 스칼라로 변환
    # return avg_mse + total_kl, avg_snr
    return avg_mse, avg_snr

@torch.no_grad()
def mlloss(noisy: torch.Tensor, enhanced: torch.Tensor, clean: torch.Tensor):
    # PESQ wrapper는 CPU tensor 입력을 기대하므로 detach→cpu
    noisy_1d    = noisy.detach().cpu().squeeze(1)      # [B, T]
    enhanced_1d = enhanced.detach().cpu().squeeze(1)   # [B, T]
    clean_1d = clean.detach().cpu().squeeze(1)  # [B, T]

    orig_pesq = pesq(noisy_1d,    clean_1d)    # [B]
    enh_pesq = pesq(enhanced_1d, clean_1d)  # [B]

    reward = torch.clamp(enh_pesq - orig_pesq, min=0.0)  # [B]
    return reward.to(clean.device)