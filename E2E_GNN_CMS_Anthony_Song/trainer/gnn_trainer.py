import copy
import torch
import wandb
from tqdm.auto import tqdm

from utils.train_utils import AverageMeter

m0_scale    = 85
m1_scale    = 415

class GNNTrainer():
    def __init__(self, num_epochs, model, mode, criterion, optimizer, scheduler, train_loader, train_batch_size, train_size, val_loader, val_batch_size, val_size, device, multi_gpu, **kwargs) -> None:
        self.num_epochs = num_epochs
        self.model = model
        self.mode = mode
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.train_batch_size = train_batch_size
        self.train_size = train_size
        self.val_loader = val_loader
        self.val_batch_size = val_batch_size
        self.val_size = val_size
        self.device = device
        self.multi_gpu = multi_gpu
        self.args = kwargs

    def train(self):

        best_model = copy.deepcopy(self.model).to("cpu", non_blocking=True)
        best_val_loss = float("inf")
        val_loss_avg_meter = AverageMeter()
        val_regress_loss_avg_meter = AverageMeter()
        val_mae_avg_meter = AverageMeter()
        val_class_loss_avg_meter = AverageMeter()

        metric = torch.nn.L1Loss()

        for epoch in range(self.num_epochs):
            self.model.train()
            tqdm_iter = tqdm(self.train_loader, total=len(self.train_loader))
            tqdm_iter.set_description(f"Epoch {epoch}")

            for it, batch in enumerate(tqdm_iter):
                self.optimizer.zero_grad()
                
                if not self.multi_gpu:
                    batch = batch.to(self.device, non_blocking=True)
                    m = batch.y
                else:
                    m = torch.cat([data.y.unsqueeze(-1) for data in batch]).to(self.device)

                out = self.model(batch)

                loss_dict = {}
                loss = 0

                if self.mode == "regression":

                    loss_dict['regress'] = self.criterion['regress'](out['regress'], m.unsqueeze(-1))
                    loss += loss_dict['regress']

                    if self.args["output_norm_scaling"]:
                        m = m * self.args["output_norm_value"]
                        out['regress'] = out['regress'] * self.args["output_norm_value"]
                    elif self.args["scale_histogram"]:
                        m = m * m1_scale
                        out['regress'] = out['regress'] * m1_scale

                    if self.args["output_mean_scaling"]:
                        m = m + self.args["output_mean_value"]
                        out['regress'] = out['regress'] + self.args["output_mean_value"]
                    elif self.args["scale_histogram"]:
                        m = m + m0_scale
                        out['regress'] = out['regress'] + m0_scale

                    mae = metric(out['regress'].detach(), m.unsqueeze(-1))

                    postfix_dict = {
                        'regress_loss': loss_dict['regress'].item(),
                        'mae': mae.item(),           
                    }

                    if self.args["predict_bins"]:
                        if not self.multi_gpu:
                            y_class = batch.y_class
                        else:
                            y_class = torch.cat([torch.as_tensor(data.y_class).unsqueeze(-1) for data in batch]).to(self.device)
                        loss_dict['class'] = self.criterion['class'](out['class'], y_class)
                        loss += loss_dict['class']
                        postfix_dict['class_loss'] = loss_dict['class'].item()

                    postfix_dict['loss'] = loss.item()
                    
                    tqdm_iter.set_postfix(postfix_dict)
                    if not self.args["debug"]:
                        wandb_dict = {
                            "train_regress_loss": loss_dict['regress'].item(),
                            "train_loss": loss.item(),
                            "train_mae": mae.item(),
                            "train_step": (it * self.train_batch_size) + epoch * self.train_size,
                        }
                        if self.args["predict_bins"]:
                            wandb_dict['train_class_loss'] = loss_dict['class'].item()
                        wandb.log(
                            wandb_dict
                        )
                elif self.mode == "classification":
                    postfix_dict = {}
                    if self.args["predict_bins"]:
                        if not self.multi_gpu:
                            y_class = (batch.y_class).float()
                        else:
                            y_class = torch.cat([torch.as_tensor(data.y_class).unsqueeze(-1) for data in batch]).to(self.device)
                        loss_dict['class'] = self.criterion['class'](out['class'], y_class)
                        loss += loss_dict['class']
                        postfix_dict['class_loss'] = loss_dict['class'].item()

                    postfix_dict['loss'] = loss.item()
                    
                    tqdm_iter.set_postfix(postfix_dict)
                    if not self.args["debug"]:
                        wandb_dict = {
                            "train_loss": loss.item(),
                            "train_step": (it * self.train_batch_size) + epoch * self.train_size,
                        }
                        if self.args["predict_bins"]:
                            wandb_dict['train_class_loss'] = loss_dict['class'].item()
                        wandb.log(
                            wandb_dict
                        )








                loss.backward()
                self.optimizer.step()

                # if self.args["plot"] and (it * self.train_batch_size) % 1000 == 0:
                #     plot_(out, m, str((it * self.train_batch_size) + epoch * self.train_size))






            self.model.eval()
            val_tqdm_iter = tqdm(self.val_loader, total=len(self.val_loader))
            val_tqdm_iter.set_description(f"Validation Epoch {epoch}")
            val_loss_avg_meter.reset()
            val_regress_loss_avg_meter.reset()
            val_class_loss_avg_meter.reset()
            val_mae_avg_meter.reset()

            for it, batch in enumerate(val_tqdm_iter):
                with torch.no_grad():
                    
                    if not self.multi_gpu:
                        batch = batch.to(self.device, non_blocking=True)
                        m = batch.y
                    else:
                        m = torch.cat([data.y.unsqueeze(-1) for data in batch]).to(self.device)

                    out = self.model(batch)

                    loss_dict = {}
                    loss = 0

                    if self.mode == "regression":    
                        loss_dict['regress'] = self.criterion['regress'](out['regress'], m.unsqueeze(-1))
                        loss += loss_dict['regress']

                        if self.args["output_norm_scaling"]:
                            m = m * self.args["output_norm_value"]
                            out['regress'] = out['regress'] * self.args["output_norm_value"]
                        elif self.args["scale_histogram"]:
                            m = m * m1_scale
                            out['regress'] = out['regress'] * m1_scale

                        if self.args["output_mean_scaling"]:
                            m = m + self.args["output_mean_value"]
                            out['regress'] = out['regress'] + self.args["output_mean_value"]
                        elif self.args["scale_histogram"]:
                            m = m + m0_scale
                            out['regress'] = out['regress'] + m0_scale

                        mae = metric(out['regress'], m.unsqueeze(-1))
                        val_mae_avg_meter.update(mae.item(), out['regress'].size(0))
                        val_regress_loss_avg_meter.update(loss_dict['regress'].item(), out['regress'].size(0))

                        postfix_dict = {
                            'regress_loss': loss_dict['regress'].item(),
                            'mae': mae.item(),
                            'avg_mae': val_mae_avg_meter.avg,
                        }

                        if self.args["predict_bins"]:
                            if not self.multi_gpu:
                                y_class = batch.y_class
                            else:
                                y_class = torch.cat([torch.as_tensor(data.y_class).unsqueeze(-1) for data in batch]).to(self.device)
                            loss_dict['class'] = self.criterion['class'](out['class'], y_class)
                            loss += loss_dict['class']
                            postfix_dict['class_loss'] = loss_dict['class'].item()
                            val_class_loss_avg_meter.update(loss_dict['class'].item(), out['class'].size(0))
                        
                        val_loss_avg_meter.update(loss.item(), out['regress'].size(0))
                        postfix_dict['loss'] = loss.item()
                        postfix_dict['avg_loss'] = val_loss_avg_meter.avg

                        val_tqdm_iter.set_postfix(postfix_dict)
                        if not self.args["debug"]:
                            wandb_dict = {
                                "val_loss": loss.item(),
                                "val_regress_loss": loss_dict['regress'].item(),
                                "val_mae": mae.item(),
                                "val_step": (it * self.val_batch_size) + epoch * self.val_size,
                            }
                            if self.args["predict_bins"]:
                                wandb_dict['val_class_loss'] = loss_dict['class'].item()

                            wandb.log(
                                wandb_dict
                            )

                    elif self.mode == "classification":
                        postfix_dict = {}
                        if self.args["predict_bins"]:
                            if not self.multi_gpu:
                                y_class = batch.y_class.float()
                            else:
                                y_class = torch.cat([torch.as_tensor(data.y_class).unsqueeze(-1) for data in batch]).to(self.device)
                            loss_dict['class'] = self.criterion['class'](out['class'], y_class)
                            loss += loss_dict['class']
                            postfix_dict['class_loss'] = loss_dict['class'].item()
                            val_class_loss_avg_meter.update(loss_dict['class'].item(), out['class'].size(0))
                        
                        postfix_dict['loss'] = loss.item()


                        val_tqdm_iter.set_postfix(postfix_dict)
                        if not self.args["debug"]:
                            wandb_dict = {
                                "val_loss": loss.item(),
                                "val_step": (it * self.val_batch_size) + epoch * self.val_size,
                            }
                            if self.args["predict_bins"]:
                                wandb_dict['val_class_loss'] = loss_dict['class'].item()

                            wandb.log(
                                wandb_dict
                            )

            if not self.args["debug"]:
                wandb_dict = {
                    "avg_val_loss": val_loss_avg_meter.avg,
                    "avg_val_mae": val_mae_avg_meter.avg,
                    "avg_val_regress_loss": val_regress_loss_avg_meter.avg,
                    "val_epoch": epoch
                }
                wandb_dict['lr'] = self.optimizer.param_groups[0]['lr']
                if self.args["predict_bins"]:
                    wandb_dict['avg_val_class_loss'] = val_class_loss_avg_meter.avg

                wandb.log(
                    wandb_dict
                )
            if val_loss_avg_meter.avg < best_val_loss:
                best_model = copy.deepcopy(self.model).to("cpu", non_blocking=True)
                best_val_loss = val_loss_avg_meter.avg

            self.scheduler.step()

        del self.model

        return best_model.to(self.device, non_blocking=True)

