import argparse
import collections
import torch
import numpy as np
import wandb
import preprocessor.preprocessor as module_preprocessor
import data_loader.data_loaders as module_data
import models.loss_loader as module_loss
import models.metric_loader as module_metric
import models.model_loader as module_arch
import models.optimizer_loader as module_optimizer
import trainer.gnn_trainer as module_trainer
from parse_config import ConfigParser
from trainer.trainer import Trainer
from utils import prepare_device


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    if config['name'] == 'graph_regression':
        if not config["debug"]:
            wandb.init(name=config["name"], project="gsoc-gnn-runs")
            wandb.config.update(config)

        if config["device"] == 'cuda':
            if not torch.cuda.is_available():
                print("WARNING: CUDA is not available on the device. Falling back to CPU")
                config["device"] = 'cpu'

        logger = config.get_logger('train')

        # setup preprocessor instances
        preprocessor = config.init_obj('preprocessor', module_preprocessor)

        # setup data_loader instances
        data_loader = config.init_obj('data_loader', module_data, train_dset = preprocessor.get_train_dataset(), val_dset= preprocessor.get_val_dataset(), test_dset = preprocessor.get_test_dataset())
        train_loader, val_loader, test_loader = data_loader.get_data_loader()
        # valid_data_loader = data_loader.split_validation()

        # build model architecture, then print to console
        model_loader = config.init_obj('model_loader', module_arch, train_loader = train_loader)
        model = model_loader.get_model()
        logger.info(model)




        # prepare for (multi-device) GPU training
        # device, device_ids = prepare_device(config['n_gpu'])
        # model = model.to(device)
        # if len(device_ids) > 1:
        #     model = torch.nn.DataParallel(model, device_ids=device_ids)




        # get function handles of loss and metrics
        # criterion = getattr(module_loss, config['loss'])
        loss_loader = config.init_obj('loss_loader',module_loss)
        criterion = loss_loader.get_loss()


        # metrics = [getattr(module_metric, met) for met in config['metrics']]
        metric_loader = config.init_obj('metric_loader',module_metric)
        metric = metric_loader.get_metric()


        # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
        # trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        # optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
        # lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer_loader = config.init_obj('optimizer_loader',module_optimizer,trainable_params=trainable_params)
        optimizer, scheduler = optimizer_loader.get_optimizer()

        trainer = config.init_obj('trainer', module_trainer,
            model = model, 
            criterion = criterion, 
            optimizer = optimizer, 
            scheduler = scheduler, 
            train_loader = train_loader, 
            train_size = len(train_loader), 
            val_loader = val_loader, 
            val_size = len(val_loader)
            )
        
        trainer.train()

    else:
        from trainer.gnn_classification_trainer import train_graph_classifier
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # setup preprocessor instances
        preprocessor = config.init_obj('preprocessor', module_preprocessor)

        # setup data_loader instances
        data_loader = config.init_obj('data_loader', module_data, train_dset = preprocessor.get_train_dataset(), val_dset= preprocessor.get_val_dataset(), test_dset = preprocessor.get_test_dataset())
        train_loader, val_loader, test_loader = data_loader.get_data_loader()

        model, result = train_graph_classifier(model_name="GCN", device = device, train_loader =train_loader ,val_loader = val_loader, test_loader = test_loader, c_in=3, c_hidden=64, c_out=1)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
