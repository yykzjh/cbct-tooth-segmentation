# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/12/1 22:50
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import random

import numpy as np

from .elastic_deform import ElasticTransform
from .random_crop import RandomCropToLabels
from .random_flip import RandomFlip
from .random_rescale import RandomRescale
from .random_rotate import RandomRotation
from .random_shift import RandomShift
from .gaussian_noise import GaussianNoise
from .to_tensor import ToTensor
from .normalize import Normalize
from .clip_and_shift import ClipAndShift


class ComposeTransforms(object):
    """
    串行执行一系列transforms
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, label=None):
        if label is None:
            for transform in self.transforms:
                img = transform(img)
            return img
        else:
            for transform in self.transforms:
                img, label = transform(img, label)
            return img, label


class RandomAugmentChoice(object):
    """
    choose a random tranform from list an apply
    transforms: tranforms to apply
    p: probability
    """

    def __init__(self, transforms=[],
                 p=0.5):
        self.transforms = transforms
        self.p = p

    def __call__(self, img_tensor, label=None):
        augment = np.random.random(1) < self.p

        if label is None:
            if not augment:
                return img_tensor
            t = random.choice(self.transforms)
            return t(img_tensor)
        else:
            if not augment:
                return img_tensor, label
            t = random.choice(self.transforms)
            return t(img_tensor, label)


class RandomAugmentCompose(object):

    def __init__(self, transforms=[], p=0.9):
        self.transforms = transforms
        self.p = p

    def __call__(self, img_tensor, label=None):
        augment = np.random.random(1) < self.p

        if label is None:
            if not augment:
                return img_tensor
            for t in self.transforms:
                img_tensor = t(img_tensor)
            return img_tensor
        else:
            if not augment:
                return img_tensor, label
            for t in self.transforms:
                img_tensor, label = t(img_tensor, label)
            return img_tensor, label
