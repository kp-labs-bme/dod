import numpy as np
from torch.optim.lr_scheduler import _LRScheduler

class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, T_max, warmup_epochs, eta_min=0, eta_max=0.1, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.T_max = T_max
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            warmup_stage = self.last_epoch / self.warmup_epochs
            return [base_lr + (self.eta_max - base_lr) * warmup_stage for base_lr in self.base_lrs]
        else:
            # Cosine annealing after warmup
            mid_stage = (self.last_epoch - self.warmup_epochs) / (self.T_max)
            cosine_value = np.cos(np.pi * mid_stage)
            return [
                self.eta_min + (self.eta_max - self.eta_min) * (1 + cosine_value) / 2
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.optimizer.step()
        self._last_lr = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, self._last_lr):
            param_group['lr'] = lr
