# ---------------------------------------------------------------------------- #
#                                 "lib" module
#
# Library Name: augmentlib
# Author: Lorenzo Federici
# Creation Date: October 5, 2023
# Description: This library contains a set of useful tools for data augmentation
# Project Name: LUS-MULTITASK-LEARNING
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
#                                    Imports                                   #
# ---------------------------------------------------------------------------- #
import random
import torch
import torchvision.transforms as transforms

# ---------------------------------------------------------------------------- #
#                                    Methods                                   #
# ---------------------------------------------------------------------------- #
# Define a custom transformation for elastic wrapping
class ElasticWarpping(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, data):
        frame, mask = data

        if torch.rand(1) < 0.5:
            elastic_transform = transforms.ElasticTransform(alpha=self.alpha)
            frame = elastic_transform(frame)
            mask  = elastic_transform(mask)
        return frame, mask

# Define a custom transformation for cropping (applied to both frame and mask)
class Cropping(object):
    def __init__(self, crop_size):
        self.crop_size = (crop_size, crop_size)

    def __call__(self, data):
        frame, mask = data

        if torch.rand(1) < 0.5:
            five_crop = transforms.FiveCrop(size=self.crop_size)
            frame_cropped = five_crop(frame)
            mask_cropped  = five_crop(mask)

            i = random.randint(0, 4)
            frame = frame_cropped[i]
            mask = mask_cropped[i]

        return frame, mask

# Define a custom transformation for blur
class Blur(object):
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def __call__(self, data):
        frame, mask = data

        # Apply blur exclusively to the frame
        if self.kernel_size > 0 and torch.rand(1) < 0.5:
            frame_tensor = transforms.functional.gaussian_blur(frame, kernel_size=self.kernel_size)

        return frame, mask

# Define a custom transformation for random rotation
class RandomRotation(object):
    def __init__(self, rotation_degree):
        self.degree = rotation_degree

    def __call__(self, data):
        frame, mask = data

        if torch.rand(1) < 0.5:
            # Apply the same random rotation to both frame and mask
            angle = random.uniform(-self.degree, self.degree)
            frame = transforms.functional.rotate(frame, angle)
            mask  = transforms.functional.rotate(mask, angle)

        return frame, mask


# Define a custom transformation for ColorJitter
class AdjustContrast(object):
    def __init__(self, contrast_factor):
        self.contrast_factor = contrast_factor

    def __call__(self, data):
        frame, mask = data

        # Apply jitter exclusively to the frame
        if torch.rand(1) < 0.5:
            frame = transforms.functional.adjust_contrast(frame, self.contrast_factor)
        return frame, mask
