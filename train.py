import os
import argparse
import logging

import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_loader.dataloader import AudioDataset
from model.modules import Base_model
import trainingmodule
from utils.util import update_namespace_from_yaml
import wandb


def setup_logging():
    """기본 로그 설정: single-process이므로 INFO 레벨로 고정"""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s ▸ %(message)s",
        datefmt="%H:%M:%S"
    )
    return logging.getLogger("SE-RL")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = setup_logging()

    # WandB (single-process이므로 rank 개념 없음)
    wandb_run = wandb.init(
        project="SE-RL",
        name=args.experiment_name,
        entity="do-hyeon-gwangju-institute-of-science-and-technology",
        config=vars(args)
    )

    # 데이터셋 & DataLoader
    train_dataset = AudioDataset(
        noisy_root=args.train_noisy_data_path,
        clean_root=args.train_clean_data_path,
        list_file=args.train_file,
        sample_rate=args.sample_rate
    )
    val_dataset = AudioDataset(
        noisy_root=args.valid_noisy_data_path,
        clean_root=args.valid_clean_data_path,
        list_file=args.valid_file,
        sample_rate=args.sample_rate
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # 모델 생성 및 DataParallel 래핑
    model = Base_model(
        in_nc=args.in_nc,
        out_nc=args.out_nc,
        nf=args.nf,
        gc=args.ns,
        times=args.times,
        normalize=args.normalize
    ).to(device)

    # Optimizer & Scheduler 설정
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.l2
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.scheduler_min_lr,
        verbose=True
    )

    # Trainer 실행
    trainer = trainingmodule.Trainer(
        train_dataloader,
        val_dataloader,
        model,
        optimizer,
        scheduler,
        opt=args,
        wandb_run=wandb_run
    )
    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SE-RL DDP trainer")
    parser.add_argument("--config", "-c", type=str, default="config/train_config.yaml",
                        help="YAML config file")
    parser.add_argument("--resume_path", "-r", type=str, default="",
                        help="Checkpoint to resume (overrides YAML)")
    args = parser.parse_args()
    # CLI에서 받은 resume_path 따로 저장
    cli_resume_path = args.resume_path

    # YAML 덮어쓰기
    args = update_namespace_from_yaml(args, args.config)

    # CLI 값이 있으면 다시 우선 적용
    if cli_resume_path:
        args.resume_path = cli_resume_path

    main(args)

# torchrun --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29500 python train.py --config configs/train_config.yaml