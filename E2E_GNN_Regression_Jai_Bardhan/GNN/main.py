from dataset_utils import get_loaders
from dataset import get_datasets
import wandb
import torch
from model import get_model
from train_utils import get_criterion, get_optimizer, get_test_metric
from trainer import train
from tester import test
import argparse
import os


def save_model(model, save_path):
    '''
        Saves the model to the path provided.
        Args:
            model: The model weights to be saved
            save_path: The root dir path to save the model in
        
        Returns:
            None
    '''
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, "model.pth"))


def main(
    args,
    run_name,
    num_epochs,

    model,
    criterion,
    test_metric,

    optimizer,
    scheduler,

    train_loader,
    val_loader,
    test_loader,
    train_size,
    val_size,
    test_size,
    device,
    multi_gpu,

    save_path
):
    ''''
        Runs the training and testing of the model given the arguments and the configurations.
        Args:
            args: The Argparse parsed arguments
            run_name: The name to log the run on WandB
            num_epochs: The number of epochs to run the training for
            model: The model to train
            criterion: The criterion to use for training and validation
            test_metric: The metric for evaluating the test results
            optimizer: The optimizer of the model
            scheduler: The epoch based scheduler for the optimizer
            train_loader: The train dataset data loader
            val_loader: The validation dataset data loader
            test_loader: The test dataset data loader
            train_size: The size of the training dataset
            val_size: The size of the validation dataset
            test_size: The size of the test dataset
            device: The device to run the training on
            multi_gpu: Whether to run the training/testing on multiple gpus
            save_path: Path on the disk to the folder where to save the best model

        Returns:
            The best model
    '''
    if not args.debug:
        wandb.init(name=run_name, project="gsoc-gnn-runs")
        wandb.config.update(args)

    if device == 'cuda':
        if not torch.cuda.is_available():
            print("WARNING: CUDA is not available on the device. Falling back to CPU")
            device = 'cpu'

    model = train(args, num_epochs, model, criterion, optimizer, scheduler,
                  train_loader, args.train_batch_size, train_size,
                  val_loader, args.val_batch_size, val_size, device, multi_gpu)

    test_error, results = test(args, model, test_loader, test_metric, device, output_norm_scaling=args.output_norm_scaling,
                      output_norm_value=args.output_norm_value, results_to_get=[])
    print(f"Model on Test dataset - Error: {test_error}")

    if not args.debug:
        wandb.log({
            "test_error": test_error
        })

        wandb.finish()

    save_model(model, save_path)

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='Number of epochs to train')
    parser.add_argument('--device', type=str, choices=[
                        'cpu', 'cuda'], default='cuda', help='Which device to run the models on')
    parser.add_argument('--name', type=str,
                        default='cnn-model', help='Name of the run')
    parser.add_argument('--save_path', type=str,
                        default='./ckpt', help='Path to save the final model')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to the root dir of the dataset')
    parser.add_argument('--train_batch_size', type=int,
                        default=32, help='Train Batch Size')
    parser.add_argument('--val_batch_size', type=int,
                        default=32, help='Validation Batch Size')
    parser.add_argument('--test_batch_size', type=int,
                        default=32, help='Test Batch Size')
    parser.add_argument('--num_files', type=int, default=7,
                        help='Number of dataset files to load')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Ratio of the dataset to take as the test set')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='Ratio of the dataset to take as the validation set')
    parser.add_argument('--pretrained', action='store_true',
                        help='Whether to use the pretrained network')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='The learning rate of the model')
    parser.add_argument('--lr_step', type=int, default=5,
                        help='The number of steps to reduce the LR of the optimizer')
    parser.add_argument('--lr_gamma', type=float, default=0.1,
                        help='The factor by which to reduce the LR of the optimizer')
    parser.add_argument('--criterion_type', type=str, default='mse', choices=[
                        'mse', 'l2', 'l1', 'smoothl1'], help='Which criterion to use for training')
    parser.add_argument('--criterion_beta', type=float, default=20,
                        help='Beta for the specific criterion if applicable')
    parser.add_argument('--use_pe', action='store_true',
                        help='Whether to use Positional Encoding')
    parser.add_argument('--num_pe_scales', type=int,
                        default=5, help='Number of PE scales to use')
    parser.add_argument('--min_threshold', type=float, default=1e-3,
                        help='The min threshold for the zero suppression')
    parser.add_argument('--output_mean_scaling', action='store_true',
                        help='Whether to perform mean scaling on the output')
    parser.add_argument('--output_mean_value', type=float,
                        default=293.2899, help='The mean to subtract from the mean')
    parser.add_argument('--output_norm_scaling', action='store_true',
                        help='Whether to divide the output by normalizing constant')
    parser.add_argument('--output_norm_value', type=float, default=119.904,
                        help='The the normalizing constant to divide the output by')
    parser.add_argument('--model', type=str, default='gat', choices=['dgcnn', 'gat', 'pna', 'gatedgcn', 'gps'], help='The backbone GNN to use')
    parser.add_argument('--point_fn', type=str, default='total', choices=['total', 'channel_wise'], help='How to obtain points from the image')
    parser.add_argument('--plot', action='store_true', help='Whether to plot the predicted vs ground truth results')
    parser.add_argument('--edge_feat', type=str, default='none', choices=['none', 'R'], help='Which method to use to obtain edge_feat')
    parser.add_argument('--scale_histogram', action='store_true', help='Whether to scale based on histogram scales provided')
    parser.add_argument('--predict_bins', action='store_true', help='Whether to also predict a binned mass')
    parser.add_argument('--min_mass', type=float, default=0., help='Minimum mass of the output')
    parser.add_argument('--max_mass', type=float, default=1., help='Maximum mass of the output')
    parser.add_argument('--num_bins', type=int, default=10, help='Number of bins for binning the output mass')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--sched_type', type=str, default='step', choices=['step', 'ca_wm'], help='Which type of scheduler to use')
    parser.add_argument('--min_lr', type=float, default=1e-7, help='Minimum LR for the cosine annealing LR scheduler')
    parser.add_argument('--T_0', type=int, default=5, help='Number of iterations for the first restart')
    parser.add_argument('--optim', type=str, default='adam', choices=['adam', 'adamw', 'rmsprop', 'sgd'], help='Which optimizer to use')
    parser.add_argument('--LapPE', action='store_true', help='Whether to use the Laplacian PE encoding transform')
    parser.add_argument('--LapPEnorm', default=None, choices=['sym', 'rw', None])
    parser.add_argument('--LapPEmax_freq', type=int, default=10, help='Maximum number of top smallest frequencies / eigenvecs to use')
    parser.add_argument('--LapPEeig_norm', default='L2', help='Normalization for the eigen vectors of the Laplacian')
    parser.add_argument('--RWSE', action='store_true', help='Whether to perform the Random Walk Encoding Transform')
    parser.add_argument('--RWSEkernel_times', default=[2, 3, 5, 7, 10], help='List of k-steps for which to compute the RW landings')
    parser.add_argument('--save_data', action='store_true', help='Whether to store the data object for each sample')
    parser.add_argument('--gps_mpnn_type', type=str, default=None, choices=[None, 'gatedgcn', 'gat', 'edgeconv', 'pnaconv'], help='Local MPNN for the GraphGPS Layer')
    parser.add_argument('--gps_global_type', type=str, default=None, choices=[None, 'transformer', 'performer'], help='The Global Attention Module for the GraphGPS Layer')
    parser.add_argument('--gps_num_heads', type=int, default=4, help='The number of heads for the self attn of the GraphGPS layer')
    parser.add_argument('--gps_dim_h', type=int, default=128, help='The dim_h of the GraphGPS Layer')
    parser.add_argument('--num_gps_layers', type=int, default=2, help='Number of GraphGPS Layers to use')
    parser.add_argument('--multi_gpu', action='store_true', help='Whether to use multiple gpus using DataParallel')
    args = parser.parse_args()

    train_dset, val_dset, test_dset, train_size, val_size, test_size = get_datasets(
        args,
        args.data_dir,
        args.num_files,
        args.test_ratio,
        args.val_ratio,
        scale_histogram=args.scale_histogram,
        predict_bins=args.predict_bins,
        min_mass=args.min_mass,
        max_mass=args.max_mass,
        num_bins=args.num_bins,
        point_fn=args.point_fn,
        use_pe=args.use_pe,
        pe_scales=args.num_pe_scales,
        min_threshold=args.min_threshold,
        output_mean_scaling=args.output_mean_scaling,
        output_mean_value=args.output_mean_value,
        output_norm_scaling=args.output_norm_scaling,
        output_norm_value=args.output_norm_value
    )

    train_loader, val_loader, test_loader = get_loaders(
        train_dset, val_dset, test_dset, args.train_batch_size, args.val_batch_size, args.test_batch_size, args.multi_gpu)

    model = get_model(args, args.device, model=args.model, edge_feat=args.edge_feat, train_loader=train_loader, point_fn=args.point_fn, pretrained=args.pretrained,
                      use_pe=args.use_pe, pe_scales=args.num_pe_scales, predict_bins=args.predict_bins, num_bins=args.num_bins, multi_gpu=args.multi_gpu)
    optimizer, scheduler = get_optimizer(
        model, args.optim, args.lr, args.sched_type, args.lr_step, args.lr_gamma, args.min_lr, args.T_0)

    criterion = get_criterion(args.criterion_type, beta=args.criterion_beta, predict_bins=args.predict_bins)
    test_metric = get_test_metric()

    model = main(
        args=args,
        run_name=args.name,
        num_epochs=args.num_epochs,
        model=model,
        criterion=criterion,
        test_metric=test_metric,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        device=args.device,
        multi_gpu=args.multi_gpu,
        save_path=args.save_path
    )
