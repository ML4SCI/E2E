import copy
from train_utils import AverageMeter, plot_
import wandb
from tqdm.auto import tqdm
import torch

m0_scale    = 85
m1_scale    = 415

def train(args, num_epochs, model, criterion, optimizer, scheduler, train_loader, train_batch_size, train_size, val_loader, val_batch_size, val_size, device):
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
    val_mae_avg_meter = AverageMeter()

    train_loss_avg_meter = AverageMeter()
    train_mae_avg_meter = AverageMeter()

    metric = torch.nn.L1Loss()

    for epoch in range(num_epochs):
        model.train()
        tqdm_iter = tqdm(train_loader, total=len(train_loader))
        tqdm_iter.set_description(f"Epoch {epoch}")

        train_mae_avg_meter.reset()
        train_loss_avg_meter.reset()

        for it, batch in enumerate(tqdm_iter):
            optimizer.zero_grad()

            X, pt, ieta, iphi, m = (
                batch["X_jets"],
                batch["pt"],
                batch["ieta"],
                batch["iphi"],
                batch["m"],
            )

            X = X.to(device, non_blocking=True)
            pt = pt.to(device, non_blocking=True)
            ieta = ieta.to(device, non_blocking=True)
            iphi = iphi.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True)

            out = model(X, pt, ieta, iphi)

            loss = criterion(out, m.unsqueeze(-1))

            if args.output_norm_scaling:
                m = m * args.output_norm_value
                out = out * args.output_norm_value
            elif args.scale_histogram:
                m = (m * m1_scale) 
                out = out * m1_scale

            if args.output_mean_scaling:
                m = m + args.output_mean_value
                out = out + args.output_mean_value
            elif args.scale_histogram:
                m = m + m0_scale
                out = out + m0_scale

            mae = metric(out.detach(), m.unsqueeze(-1))


            train_loss_avg_meter.update(loss.item(), n=out.size(0))
            train_mae_avg_meter.update(mae.item(), n=out.size(0))
            
            tqdm_iter.set_postfix(loss=loss.item(), mae=mae.item())
            if not args.debug:
                wandb.log(
                    {
                        "train_loss": loss.item(),
                        "train_mae": mae.item(),
                        "train_step": (it * train_batch_size) + epoch * train_size,
                    }
                )

            loss.backward()
            optimizer.step()

            if args.plot and (it * train_batch_size) % 1000 == 0:
                plot_(out, m, str((it * train_batch_size) + epoch * train_size))

        if not args.debug:
            wandb_dict = {
                "avg_train_loss": train_loss_avg_meter.avg,
                "avg_train_mae": train_mae_avg_meter.avg,
                "train_epoch": epoch
            }

            wandb.log(
                wandb_dict
            )

        model.eval()
        val_tqdm_iter = tqdm(val_loader, total=len(val_loader))
        val_tqdm_iter.set_description(f"Validation Epoch {epoch}")
        val_loss_avg_meter.reset()
        val_mae_avg_meter.reset()

        for it, batch in enumerate(val_tqdm_iter):
            with torch.no_grad():
                X, pt, ieta, iphi, m = (
                    batch["X_jets"],
                    batch["pt"],
                    batch["ieta"],
                    batch["iphi"],
                    batch["m"],
                )

                X = X.to(device, non_blocking=True)
                pt = pt.to(device, non_blocking=True)
                ieta = ieta.to(device, non_blocking=True)
                iphi = iphi.to(device, non_blocking=True)
                m = m.to(device, non_blocking=True)

                out = model(X, pt, ieta, iphi)

                loss = criterion(out, m.unsqueeze(-1))

                if args.output_norm_scaling:
                    m = m * args.output_norm_value
                    out = out * args.output_norm_value
                elif args.scale_histogram:
                    m = m * m1_scale
                    out = out * m1_scale

                if args.output_mean_scaling:
                    m = m + args.output_mean_value
                    out = out + args.output_mean_value
                elif args.scale_histogram:
                    m = m + m0_scale
                    out = out + m0_scale

                mae = metric(out, m.unsqueeze(-1))

                val_loss_avg_meter.update(loss.item(), out.size(0))
                val_mae_avg_meter.update(mae.item(), out.size(0))
                val_tqdm_iter.set_postfix(loss=val_loss_avg_meter.avg, mae=val_mae_avg_meter.avg)
                if not args.debug:
                    wandb.log(
                        {
                            "val_loss": loss.item(),
                            "val_mae": mae.item(),
                            "val_step": (it * val_batch_size) + epoch * val_size,
                        }
                    )

        if not args.debug:
            wandb_dict = {
                "avg_val_loss": val_loss_avg_meter.avg,
                "avg_val_mae": val_mae_avg_meter.avg,
                "val_epoch": epoch
            }

            wandb_dict['lr'] = optimizer.param_groups[0]['lr']

            wandb.log(
                wandb_dict
            )


        if val_loss_avg_meter.avg < best_val_loss:
            best_model = copy.deepcopy(model).to("cpu", non_blocking=True)
            best_val_loss = val_loss_avg_meter.avg

        scheduler.step()

    del model

    return best_model.to(device, non_blocking=True)