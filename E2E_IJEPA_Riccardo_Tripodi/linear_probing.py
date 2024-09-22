import os

import logging
import sys
import yaml
import wandb

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import roc_auc_score

from src.utils.logging import (
    CSVLogger)

from torch.utils.data import Subset
from torch.utils.data import DataLoader
from src.datasets.ijepa_dataset import Dataset4 as ProbingDataset
from src.masks.multiblock import collate
from src.helper_probing import init_model
import src.models.vision_transformer as vit

import argparse
import pprint

parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='configs.yaml')
parser.add_argument(
    '--devices', type=str, nargs='+', default=['cuda:0'],
    help='which devices to use on local machine')

# --
log_timings = True
log_freq = 10
checkpoint_freq = 1
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def main(args, resume_preempt=False):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    model_name = args['meta']['model_name']
    pretrained_path = args['meta']['pretrained_path']
    use_batch_norm = args['meta']['use_batch_norm']
    use_hidden_layer = args['meta']['use_hidden_layer']
    num_unfreeze_layers = args['meta']['num_unfreeze_layers']
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device(args['devices'][0])
        torch.cuda.set_device(device)

    # -- DATA
    batch_size = args['data']['batch_size']
    patch_size = args['data']['patch_size']
    num_classes = args['data']['num_classes']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    root_path = args['data']['root_path']
    img_size = args['data']['img_size']
    chunk_size = args['data']['chunk_size']
    
    # -- OPTIMIZATION
    num_epochs = args['optimization']['num_epochs']
    start_lr = args['optimization']['start_lr']
    final_lr = args['optimization']['final_lr']

    # -- LOGGING
    folder = args['logging']['folder']
    folder = folder+'/Classification_'+str(args['fname'])
    if not os.path.exists(folder):
        os.makedirs(folder)
    tag = args['logging']['write_tag']

    dump = os.path.join(folder, 'params-ijepa.yaml')
    with open(dump, 'w') as f:
        yaml.dump(args, f)
    # ----------------------------------------------------------------------- #

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f'{tag}_.csv')
    epoch_log_file = os.path.join(folder, f'{tag}_epoch.csv')
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')

    # -- make csv_logger
    csv_logger = CSVLogger(log_file,
                    ('%d', 'epoch'),
                    ('%d', 'itr'),
                    ('%.5f', 'train_loss'))
    
    # -- make csv_logger for end of epoch losses
    epoch_csv_logger = CSVLogger(epoch_log_file,
                      ('%d', 'epoch'),
                      ('%.5f', 'train_loss'),
                      ('%.5f', 'val_loss'),
                      ('%.5f', 'train_acc'),
                      ('%.5f', 'val_acc'),
                      ('%.5f', 'val_auc'))

    # -- init model
    model = init_model(
        img_size=img_size,
        patch_size=patch_size,
        pretrained_path=pretrained_path,
        model_name=model_name,
        num_classes=num_classes,
        use_batch_norm=use_batch_norm,
        use_hidden_layer=use_hidden_layer,
        num_unfreeze_layers=num_unfreeze_layers
    )

    # -- count parameter
    model_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad is True)
    logger.info('Number of parameters in the classifier: %d' % model_params)
    logger.info('Trainable params: %d' % trainable_params)

    # -- init dataset
    train_dataset = ProbingDataset(file_path=root_path, mode='train', chunk_size=chunk_size)
    val_dataset = ProbingDataset(file_path=root_path, mode='validation', chunk_size=chunk_size)

    train_length = len(train_dataset)
    val_length = len(val_dataset)
    train_indices = list(range(int(train_length*95/100),train_length))
    val_indices = list(range(int(val_length*75/100),val_length))

    train_dataset = Subset(train_dataset, train_indices)
    val_dataset = Subset(val_dataset, val_indices)

    # -- init dataLoaders
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,  # Number of chunks to load in each batch
        shuffle=True,  # Shuffle the data between epochs
        collate_fn=collate,  # Use the custom collate function
        num_workers=num_workers  # Number of subprocesses to use for data loading
    )
    
    # Create the DataLoader
    val_data_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,  # Number of chunks to load in each batch
        shuffle=False,  # Shuffle the data between epochs
        collate_fn=collate,  # Use the custom collate function
        num_workers=num_workers  # Number of subprocesses to use for data loading
    )
    
    logger.info('GSOC supervised data loaders created')

    # -- init optimizer and scheduler
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=start_lr)
    loss_function = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    model = model.to(device)

    def save_checkpoint(epoch):
        save_dict = {
            'model': model.state_dict()
        }

        torch.save(save_dict, save_path)

    start_epoch = 0
    # -- Early stopping parameters
    patience = 5  
    best_val_loss = float('inf') 
    epochs_no_improvement = 0 
    delta = 0.001

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for batch_idx, data in enumerate(train_data_loader):
            inputs, train_labels = data
            padding = (0, 1, 0, 1)
            inputs = torch.nn.functional.pad(inputs, padding, mode='constant', value=0)

            inputs = inputs.float().to(device)
            train_labels = train_labels.unsqueeze(1).to(device)
            #print('train labels: ', train_labels)

            # Forward pass
            outputs = model(inputs)
            #print('outputs: ', outputs)

            # Loss computation
            loss = loss_function(outputs, train_labels)
            total_loss += loss.item()

            # Calculate predictions for accuracy
            predicted_probabilities = torch.sigmoid(outputs)
            predicted_labels = (predicted_probabilities > 0.5).float()
            #print('pred labels: ', predicted_labels)
            correct_predictions += (predicted_labels == train_labels).sum().item()
            total_predictions += train_labels.size(0)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            csv_logger.log(epoch + 1, batch_idx, loss)
            wandb.log({"Batch_loss": loss.item()})


        epoch_loss = total_loss / len(train_data_loader)
        epoch_accuracy = correct_predictions / total_predictions * 100
        print(f'Epoch {epoch+1}\nTrain loss: {epoch_loss:.4f}, Train accuracy: {epoch_accuracy:.2f}%')
        wandb.log({"Epoch_train_loss": epoch_loss})
        wandb.log({"Epoch_train_acc": epoch_accuracy})

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_correct_predictions = 0
        val_total_predictions = 0
        val_labels = []
        val_preds = []

        with torch.no_grad():
            for val_batch_idx, val_data in enumerate(val_data_loader):
                val_inputs, labels = val_data
                padding = (0, 1, 0, 1)
                val_inputs = torch.nn.functional.pad(val_inputs, padding, mode='constant', value=0)

                val_inputs = val_inputs.float().to(device)
                labels = labels.unsqueeze(1).to(device)

                # Forward pass
                logits = model(val_inputs)

                # Loss computation
                val_loss += loss_function(logits, labels).item()

                # Calculate predictions for accuracy
                val_predicted_probabilities = torch.sigmoid(logits)
                val_predicted_labels = (val_predicted_probabilities > 0.5).float()
                val_correct_predictions += (val_predicted_labels == labels).sum().item()
                val_total_predictions += labels.size(0)

                # Store labels and logits for auc
                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(torch.sigmoid(logits).cpu().numpy())

        val_epoch_loss = val_loss / len(val_data_loader)
        val_epoch_accuracy = val_correct_predictions / val_total_predictions * 100
        auc_score = roc_auc_score(val_labels, val_preds)
        print(f'Validation loss: {val_epoch_loss:.4f}, Validation accuracy: {val_epoch_accuracy:.2f}%, Val AUC score:{auc_score:.4f}')
        wandb.log({"Epoch_val_loss": val_epoch_loss})
        wandb.log({"Epoch_val_accuracy": val_epoch_accuracy})
        wandb.log({"Val_auc_score": auc_score})

        epoch_csv_logger.log(epoch + 1, epoch_loss, val_epoch_loss, epoch_accuracy, val_epoch_accuracy, auc_score)

        if best_val_loss - val_epoch_loss > delta:
            best_val_loss = val_epoch_loss
            epochs_no_improvement = 0
        else:
            epochs_no_improvement += 1

        if epochs_no_improvement >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    save_checkpoint(epoch)


if __name__ == "__main__":
    args = parser.parse_args()

    # -- load script params
    params = None
    with open(args.fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info('loaded params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)
    
    wandb.init(
        project="GSOC_classification", 
        config=params, 
        name=str(params['logging']['write_tag'])+'_'+args.fname.split('/')[-1]  # Specify the run name here
    )
    
    params['devices'] = args.devices
    params['fname'] = args.fname

    main(args=params)
    wandb.finish()
