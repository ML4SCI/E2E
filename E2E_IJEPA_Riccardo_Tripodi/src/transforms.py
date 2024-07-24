# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger

from PIL import ImageFilter

import torch
import numpy as np
import torchvision.transforms as transforms

_GLOBAL_SEED = 0
logger = getLogger()


def make_transforms(
    color_jitter=1.0,
    color_distortion=False,
    horizontal_flip=False,
    vertical_flip=False,
    gaussian_blur=False,
    gaussian_blur_std=None,
    use_rotation=False
):
    logger.info('making imagenet data transforms')

    def get_color_distortion(s=1.0):
        # s is the strength of color distortion.
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([
            rnd_color_jitter,
            rnd_gray])
        return color_distort

    transform_list = []
    if horizontal_flip:
        transform_list += [transforms.RandomHorizontalFlip()]
    if vertical_flip:
        transform_list += [transforms.RandomVerticalFlip()]
    if use_rotation:
        transform_list += [transforms.RandomRotation(30)]
    if color_distortion:
        transform_list += [get_color_distortion(s=color_jitter)]
    if gaussian_blur:
        transform_list += [AddGaussianNoiseTorch(p=0.5, std=gaussian_blur_std)]

    return transforms.Compose(transform_list)

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.prob=p
    
    def __call__(self, array):
        if np.random.rand() < self.prob:
            array = np.fliplr(array)
        return array

class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.prob=p
    
    def __call__(self, array):
        if np.random.rand() < self.prob:
            array = np.flipud(array)
        return array

class AddGaussianNoise(object):
    def __init__(self, p=0.5, std=1.0, factor=0.1):
        self.prob= p
        assert len(std) == 8
        self.std = np.array(std).reshape(-1, 1, 1)
        self.factor = factor

    def __call__(self, tensor):
        if np.random.rand() < self.prob:
            return tensor
        
        noise = np.random.randn(*tensor.shape) * (self.std * self.factor)

        return tensor + noise

class AddGaussianNoiseTorch(object):
    def __init__(self, p=0.5, std=[1.0]*8, factor=0.1):
        self.prob = p
        self.std = torch.tensor(std).view(-1, 1, 1)
        self.factor = factor

    def __call__(self, tensor):
        if torch.rand(1).item() >= self.prob:
            return tensor
        
        noise = torch.randn(tensor.size(), device=tensor.device) * (self.std * self.factor)
        return tensor + noise