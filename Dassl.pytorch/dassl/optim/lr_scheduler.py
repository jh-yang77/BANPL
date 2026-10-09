"""
Modified from https://github.com/KaiyangZhou/deep-person-reid
"""
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
AVAI_SCHEDS = ["single_step", "multi_step", "cosine"]


class _BaseWarmupScheduler(_LRScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        last_epoch=-1,
        verbose=False
    ):
        self.successor = successor
        self.warmup_epoch = warmup_epoch
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if self.last_epoch >= self.warmup_epoch:
            self.successor.step(epoch)
            self._last_lr = self.successor.get_last_lr()
        else:
            super().step(epoch)


class ConstantWarmupScheduler(_BaseWarmupScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        cons_lr,
        last_epoch=-1,
        verbose=False
    ):
        self.cons_lr = cons_lr
        super().__init__(
            optimizer, successor, warmup_epoch, last_epoch, verbose
        )

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        return [self.cons_lr for _ in self.base_lrs]


class LinearWarmupScheduler(_BaseWarmupScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        min_lr,
        last_epoch=-1,
        verbose=False
    ):
        self.min_lr = min_lr
        super().__init__(
            optimizer, successor, warmup_epoch, last_epoch, verbose
        )

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        if self.last_epoch == 0:
            return [self.min_lr for _ in self.base_lrs]
        return [
            lr * self.last_epoch / self.warmup_epoch for lr in self.base_lrs
        ]

import math
from torch.optim.lr_scheduler import LambdaLR

def build_two_stage_lr_scheduler(optimizer, total_epochs, first_stage_epochs, base_lr, second_stage_base_lr, warmup_epochs=1):
    
    # 算出第二阶段缩小多少倍 (比如 1e-3 / 3.5e-3 = 0.28)
    scale = second_stage_base_lr / base_lr
    
    def lr_lambda(epoch):
        # ============================================
        # 阶段 1: POS (Epoch 0 ~ first-1)
        # ============================================
        if epoch < first_stage_epochs:
            # 1. 热身期 (线性上升)
            if epoch < warmup_epochs:
                return float(epoch + 1) / warmup_epochs
            
            # 2. 衰减期 (Cosine 下降 1.0 -> 0.0)
            curr = epoch - warmup_epochs
            total = first_stage_epochs - warmup_epochs
            progress = curr / total
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        # ============================================
        # 阶段 2: NEG (Epoch first ~ total-1)
        # ============================================
        else:
            # 重置时间，从 0 开始算
            idx = epoch - first_stage_epochs
            duration = total_epochs - first_stage_epochs
            
            # 4. 第二次衰减 (Cosine 下降 scale -> 0.0)
            curr = idx - warmup_epochs
            total = duration - warmup_epochs
            if total <= 0: return scale # 防止除0错误
            
            progress = curr / total
            # 基础 Cosine 是 1->0，乘上 scale 变成 scale->0
            return 0.5 * (1.0 + math.cos(math.pi * progress)) * scale

    return LambdaLR(optimizer, lr_lambda)

def build_lr_scheduler(optimizer, optim_cfg):
    """A function wrapper for building a learning rate scheduler.

    Args:
        optimizer (Optimizer): an Optimizer.
        optim_cfg (CfgNode): optimization config.
    """
    lr_scheduler = optim_cfg.LR_SCHEDULER
    stepsize = optim_cfg.STEPSIZE
    gamma = optim_cfg.GAMMA
    max_epoch = optim_cfg.MAX_EPOCH

    if lr_scheduler not in AVAI_SCHEDS:
        raise ValueError(
            f"scheduler must be one of {AVAI_SCHEDS}, but got {lr_scheduler}"
        )

    if lr_scheduler == "single_step":
        if isinstance(stepsize, (list, tuple)):
            stepsize = stepsize[-1]

        if not isinstance(stepsize, int):
            raise TypeError(
                "For single_step lr_scheduler, stepsize must "
                f"be an integer, but got {type(stepsize)}"
            )

        if stepsize <= 0:
            stepsize = max_epoch

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=stepsize, gamma=gamma
        )

    elif lr_scheduler == "multi_step":
        if not isinstance(stepsize, (list, tuple)):
            raise TypeError(
                "For multi_step lr_scheduler, stepsize must "
                f"be a list, but got {type(stepsize)}"
            )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=stepsize, gamma=gamma
        )

    elif lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(max_epoch), eta_min=1e-5
        )
    if optim_cfg.WARMUP_EPOCH > 0:
        if not optim_cfg.WARMUP_RECOUNT:
            scheduler.last_epoch = optim_cfg.WARMUP_EPOCH

        if optim_cfg.WARMUP_TYPE == "constant":
            scheduler = ConstantWarmupScheduler(
                optimizer, scheduler, optim_cfg.WARMUP_EPOCH,
                optim_cfg.WARMUP_CONS_LR
            )

        elif optim_cfg.WARMUP_TYPE == "linear":
            scheduler = LinearWarmupScheduler(
                optimizer, scheduler, optim_cfg.WARMUP_EPOCH,
                optim_cfg.WARMUP_MIN_LR
            )

        else:
            raise ValueError

    return scheduler
