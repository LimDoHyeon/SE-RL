def check_parameters(model, to_mb=True):
    """
    Return total number of trainable parameters.
    If `to_mb` is True, return model size in megabytes (fp32).
    """
    tot_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if to_mb:
        # 4 bytes per fp32 parameter → 1024² for MiB
        return tot_params * 4 / 1024 / 1024
    return tot_params