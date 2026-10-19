import torch
import torch.optim as optim
from typing import Tuple
from utils.optim.lr_scheduler import WarmupCosineAnnealingLR
from factory.base_factory import BaseFactory
from utils.optim.custom_loss import FocalLoss


class AgentFactory(BaseFactory):
    @classmethod
    def create(cls, **kwargs) -> Tuple:
        config = kwargs["config"]
        model = kwargs["model"]
        loss = config.optim.loss
        optimizer_name = config.optim.optimizer
        boundaries = config.optim.boundaries
        values = config.optim.values

        if loss == "cross_entropy":
            criterion = torch.nn.CrossEntropyLoss()
        elif loss == "focal_loss":
            criterion = FocalLoss()
        else:
            raise NotImplementedError(f"Invalid loss type {loss}, can be ('cross_entropy' or 'focal_loss')")

        # Common params
        lr = config.optim.learning_rate
        momentum = config.optim.momentum
        weight_decay = config.optim.weight_decay
        eps = config.optim.eps

        if optimizer_name == "sgd":
            optimizer = optim.SGD(model.parameters(),
                                  lr=lr,
                                  momentum=momentum,
                                  weight_decay=weight_decay)
        elif optimizer_name == "adam":
            optimizer = optim.Adam(model.parameters(),
                                   lr=lr,
                                   betas=(config.optim.beta1, config.optim.beta2),
                                   eps=eps,
                                   weight_decay=weight_decay,
                                   amsgrad=config.optim.amsgrad)
        elif optimizer_name == "rmsprop":
            optimizer = optim.RMSprop(model.parameters(),
                                      lr=lr,
                                      alpha=config.optim.alpha,
                                      eps=eps,
                                      weight_decay=weight_decay,
                                      momentum=momentum,
                                      centered=config.optim.centered)
        else:
            raise NotImplementedError("Invalid optimizer type ('sgd' or 'adam' or 'rmsprop')")

        if config.optim.lr_decay:
            if config.optim.lr_decay_type == "cos":
                lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                                    T_max=config.trainer.n_epochs)
            elif config.optim.lr_decay_type == "warmup_cos":
                T_max = config.trainer.n_epochs - config.optim.warmup_epochs
                lr_scheduler = WarmupCosineAnnealingLR(optimizer=optimizer,
                                                       T_max=T_max,
                                                       warmup_epochs=config.optim.warmup_epochs,
                                                       eta_min=config.optim.lr_min,
                                                       eta_max=config.optim.warmup_lr_high)
            elif config.optim.lr_decay_type == "on_plateau":
                lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer)
            elif config.optim.lr_decay_type == "lin":
                lr_scheduler = optim.lr_scheduler.LinearLR(optimizer=optimizer)
            elif config.optim.lr_decay_type == "exp":
                lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.99)
            elif config.optim.lr_decay_type == "wstep":
                lr_scheduler = WarmupStepwiseLR(optimizer=optimizer,
                                                warmup_epochs=config.optim.warmup_epochs,
                                                boundaries=boundaries,
                                                values=values)
            else:
                raise ValueError("Invalid learning rate scheduler "
                                 "('cos' or 'warmup_cos' or 'on_plateau' or 'lin' or 'exp' or 'wstep')")
        else:
            lr_scheduler = None

        return criterion, optimizer, lr_scheduler
