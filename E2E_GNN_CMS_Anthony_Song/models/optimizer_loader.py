import torch

class OptimizerLoader():
    def __init__(self, trainable_params, optim, lr, sched_type='step', lr_step=None, lr_gamma=None, min_lr=None, T_0=None) -> None:
        self.optimizer = None
        self.scheduler = None
        self.trainable_params = trainable_params
        self.optim = optim
        self.lr = lr
        self.sched_type = sched_type
        self.lr_step = lr_step
        self.lr_gamma = lr_gamma
        self.min_lr = min_lr
        self.T_0 = T_0


    def get_optimizer(self):

        if self.optim == 'adam':
            self.optimizer = torch.optim.Adam(self.trainable_params, lr=self.lr)
        elif self.optim == 'adamw':
            self.optimizer = torch.optim.AdamW(self.trainable_params, lr=self.lr)
        elif self.optim == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(self.trainable_params, lr=self.lr)
        elif self.optim == 'sgd':
            self.optimizer = torch.optim.SGD(self.trainable_params, lr=self.lr, momentum=0.9)
        else:
            raise NotImplementedError()

        if self.sched_type == 'step':
            assert self.lr_step is not None and self.lr_gamma is not None, "lr_step and lr_gamma must be provided if sched_type is stepLR"
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=self.lr_step, gamma=self.lr_gamma)
        elif self.sched_type == 'ca_wm':
            assert self.T_0 is not None, "T_0 must be provieedd for cosine annealing LR"
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=self.T_0, eta_min=self.min_lr, T_mult=2)

        return self.optimizer, self.scheduler

