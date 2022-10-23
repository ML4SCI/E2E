import copy
from train_utils import AverageMeter, plot_
import wandb
from tqdm.auto import tqdm
import torch

m0_scale    = 85
m1_scale    = 415

def train(args, num_epochs, model, criterion, optimizer, scheduler, train_loader, train_batch_size, train_size, val_loader, val_batch_size, val_size, device, multi_gpu):
    '''
        Performs the training of the model, logs the results on Wandb and returns the best model.
        Args:
            args: The Argparse parsed arguments
            num_epochs: The number of epochs to train the model for
            model: The input model to train
            criterion: The criterion to generate the losses for the model
            optimizer: The optimizer for the model
            scheduler: The epoch-based scheduler for the model
            train_loader: The training dataset data loader
            train_batch_size: The batch size of the train loader
            train_size: The total size of the training dataset
            val_loader: The validation dataset data loader
            val_batch_size: The batch size of the val loader
            val_size: The total size of the validation dataset
            device: The device to run the training on


        Returns:
            best_model: The model with the best validation loss 
    '''
    best_model = copy.deepcopy(model).to("cpu", non_blocking=True)
    best_val_loss = float("inf")
    val_loss_avg_meter = AverageMeter()
    val_regress_loss_avg_meter = AverageMeter()
    val_mae_avg_meter = AverageMeter()
    val_class_loss_avg_meter = AverageMeter()

    metric = torch.nn.L1Loss()

    for epoch in range(num_epochs):
        model.train()
        tqdm_iter = tqdm(train_loader, total=len(train_loader))
        tqdm_iter.set_description(f"Epoch {epoch}")

        for it, batch in enumerate(tqdm_iter):
            optimizer.zero_grad()
            
            if not multi_gpu:
                batch = batch.to(device, non_blocking=True)
                m = batch.y
            else:
                m = torch.cat([data.y.unsqueeze(-1) for data in batch]).to(device)

            out = model(batch)

            loss_dict = {}
            loss = 0

            loss_dict['regress'] = criterion['regress'](out['regress'], m.unsqueeze(-1))
            loss += loss_dict['regress']

            if args.output_norm_scaling:
                m = m * args.output_norm_value
                out['regress'] = out['regress'] * args.output_norm_value
            elif args.scale_histogram:
                m = m * m1_scale
                out['regress'] = out['regress'] * m1_scale

            if args.output_mean_scaling:
                m = m + args.output_mean_value
                out['regress'] = out['regress'] + args.output_mean_value
            elif args.scale_histogram:
                m = m + m0_scale
                out['regress'] = out['regress'] + m0_scale

            mae = metric(out['regress'].detach(), m.unsqueeze(-1))

            postfix_dict = {
                'regress_loss': loss_dict['regress'].item(),
                'mae': mae.item(),           
            }

            if args.predict_bins:
                if not multi_gpu:
                    y_class = batch.y_class
                else:
                    y_class = torch.cat([torch.as_tensor(data.y_class).unsqueeze(-1) for data in batch]).to(device)
                loss_dict['class'] = criterion['class'](out['class'], y_class)
                loss += loss_dict['class']
                postfix_dict['class_loss'] = loss_dict['class'].item()

            postfix_dict['loss'] = loss.item()
            
            tqdm_iter.set_postfix(postfix_dict)
            if not args.debug:
                wandb_dict = {
                    "train_regress_loss": loss_dict['regress'].item(),
                    "train_loss": loss.item(),
                    "train_mae": mae.item(),
                    "train_step": (it * train_batch_size) + epoch * train_size,
                }
                if args.predict_bins:
                    wandb_dict['train_class_loss'] = loss_dict['class'].item()
                wandb.log(
                    wandb_dict
                )

            loss.backward()
            optimizer.step()

            if args.plot and (it * train_batch_size) % 1000 == 0:
                plot_(out, m, str((it * train_batch_size) + epoch * train_size))

        model.eval()
        val_tqdm_iter = tqdm(val_loader, total=len(val_loader))
        val_tqdm_iter.set_description(f"Validation Epoch {epoch}")
        val_loss_avg_meter.reset()
        val_regress_loss_avg_meter.reset()
        val_class_loss_avg_meter.reset()
        val_mae_avg_meter.reset()

        for it, batch in enumerate(val_tqdm_iter):
            with torch.no_grad():
                
                if not multi_gpu:
                    batch = batch.to(device, non_blocking=True)
                    m = batch.y
                else:
                    m = torch.cat([data.y.unsqueeze(-1) for data in batch]).to(device)

                out = model(batch)

                loss_dict = {}
                loss = 0

                loss_dict['regress'] = criterion['regress'](out['regress'], m.unsqueeze(-1))
                loss += loss_dict['regress']

                if args.output_norm_scaling:
                    m = m * args.output_norm_value
                    out['regress'] = out['regress'] * args.output_norm_value
                elif args.scale_histogram:
                    m = m * m1_scale
                    out['regress'] = out['regress'] * m1_scale

                if args.output_mean_scaling:
                    m = m + args.output_mean_value
                    out['regress'] = out['regress'] + args.output_mean_value
                elif args.scale_histogram:
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

                if args.predict_bins:
                    if not multi_gpu:
                        y_class = batch.y_class
                    else:
                        y_class = torch.cat([torch.as_tensor(data.y_class).unsqueeze(-1) for data in batch]).to(device)
                    loss_dict['class'] = criterion['class'](out['class'], y_class)
                    loss += loss_dict['class']
                    postfix_dict['class_loss'] = loss_dict['class'].item()
                    val_class_loss_avg_meter.update(loss_dict['class'].item(), out['class'].size(0))
                
                val_loss_avg_meter.update(loss.item(), out['regress'].size(0))
                postfix_dict['loss'] = loss.item()
                postfix_dict['avg_loss'] = val_loss_avg_meter.avg

                val_tqdm_iter.set_postfix(postfix_dict)
                if not args.debug:
                    wandb_dict = {
                        "val_loss": loss.item(),
                        "val_regress_loss": loss_dict['regress'].item(),
                        "val_mae": mae.item(),
                        "val_step": (it * val_batch_size) + epoch * val_size,
                    }
                    if args.predict_bins:
                        wandb_dict['val_class_loss'] = loss_dict['class'].item()

                    wandb.log(
                        wandb_dict
                    )
        if not args.debug:
            wandb_dict = {
                "avg_val_loss": val_loss_avg_meter.avg,
                "avg_val_mae": val_mae_avg_meter.avg,
                "avg_val_regress_loss": val_regress_loss_avg_meter.avg,
                "val_epoch": epoch
            }
            wandb_dict['lr'] = optimizer.param_groups[0]['lr']
            if args.predict_bins:
                wandb_dict['avg_val_class_loss'] = val_class_loss_avg_meter.avg

            wandb.log(
                wandb_dict
            )
        if val_loss_avg_meter.avg < best_val_loss:
            best_model = copy.deepcopy(model).to("cpu", non_blocking=True)
            best_val_loss = val_loss_avg_meter.avg

        scheduler.step()

    del model

    return best_model.to(device, non_blocking=True)