from numpy import require
import torch
import torchvision.transforms as T
import numpy as np


def get_transforms():
    '''
        Returns the required transformation to be applied on the dataset.
        
        Args:
            None

        Returns:
            The list of transforms: List[torchvison.Transforms]
    '''

    required_transform = [
        # T.Resize(224),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        # T.Normalize(mean=[0.5] * 8,
        # std=[0.5] * 8)
        # T.Normalize(mean=[0.01037084, 0.0103173, 0.01052679, 0.01034378, 0.01097225, 0.01024814, 0.01037642, 0.01058754],
        #             std=[10.278656283775618, 7.64753320751208, 16.912319597559645, 9.005579923580713, 21.367327333103688, 7.489890622699373, 12.977402491253788, 24.50774893130742])
        # T.Normalize(mean=[0.485, 0.456, 0.406],
        #             std=[0.229, 0.224, 0.225]),
        # T.RandomAdjustSharpness(0.5, p=0.1),
    ]
    return required_transform


def get_loaders(train_dset, val_dset, test_dset, train_batch_size, val_batch_size, test_batch_size):
    '''
        This function provides the loaders for the datasets
        
        Args:
            train_dset: Training dataset
            val_dset: Validation dataset
            test_dset: Test dataset
            train_batch_size: Training batch size
            val_batch_size: Validation batch size
            test_batch_size: Test batch size

        Returns:
            train_loader: Training dataset data loader
            val_loader: Validation dataset data loader
            test_loader: Test dataset data loader
    '''
    train_loader = torch.utils.data.DataLoader(
        train_dset, shuffle=True, batch_size=train_batch_size, pin_memory=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dset, shuffle=False, batch_size=val_batch_size, pin_memory=True, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_dset, shuffle=False, batch_size=test_batch_size, num_workers=0
    )
    return train_loader, val_loader, test_loader


def positional_encoding(data, pe_scales):
    '''
        Performs cos/sin positional encoding provided the data and the scales
        Args:
            data: The data on which the positional encoding has to be applied
            pe_scales: The scales of the positional encoding

        Returns:
            The tensor with the cos/sin positional encoding performed on the data 
    '''
    pe_cos = torch.cat([torch.cos(2**i * np.pi * torch.as_tensor(data))
                       for i in range(pe_scales)])
    pe_sin = torch.cat([torch.sin(2**i * np.pi * torch.as_tensor(data))
                       for i in range(pe_scales)])

    return torch.cat([data, pe_cos, pe_sin])


def zero_suppression(X_jets, min_threshold):
    '''
        Performs zero suppression on the image
        Args:
            X_jets: The image on which zero suppression needs to be performed
            min_threshold: The threshold below which to set the pixel value as 0

        Returns:
            The zero suppressed image
    '''
    return np.where(np.abs(X_jets) >= min_threshold, X_jets, 0.)
