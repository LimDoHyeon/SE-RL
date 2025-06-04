import yaml
from argparse import Namespace
from typing import Any, Mapping


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


# 기존에 있던 함수 (check_parameters 등) 그대로 두고, 아래 함수만 추가합니다.
def update_namespace_from_yaml(args: Namespace, cfg_path: str,
                               strict: bool = False) -> Namespace:
    """
    YAML 설정 파일을 읽어 argparse.Namespace 객체에 덮어씌웁니다.

    Parameters
    ----------
    args : argparse.Namespace
        `parser.parse_args()` 로 얻은 객체. 최소한 `config` 속성만 존재하면 됩니다.
    cfg_path : str
        YAML 설정 파일 경로.
    strict : bool, default False
        True 로 설정하면 YAML 키가 Namespace 에 존재하지 않을 때 오류를 발생시킵니다.

    Returns
    -------
    argparse.Namespace
        YAML 내용이 반영된 Namespace
    """
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg: Mapping[str, Any] = yaml.safe_load(f)

    for k, v in cfg.items():
        if strict and not hasattr(args, k):
            raise KeyError(f"Unknown config key: {k}")
        setattr(args, k, v)

    return args