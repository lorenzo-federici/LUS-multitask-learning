# ---------------------------------------------------------------------------- #
#                                 "lib" module
#
# Library Name: visual_lib
# Author: Lorenzo Federici
# Creation Date: September 21, 2023
# Description: This library contains a set of useful tools for frame visualization
# Project Name: LUS-MULTITASK-LEARNING
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
#                                    Imports                                   #
# ---------------------------------------------------------------------------- #

import os
import io
import json
import base64
import cv2

import PIL
from PIL import Image
from PIL import ImageDraw

import h5py

import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
import uuid

from tqdm import tqdm

from pathlib import Path
from pathlib import PosixPath

import random
import pickle

from collections import defaultdict

import torch
from torch.utils.data import Dataset, Subset, DataLoader

# ---------------------------------------------------------------------------- #
#                                    Methods                                   #
# ---------------------------------------------------------------------------- #

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Function For Printing Frame~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_RGB_mask_color(target_array):
    color_mapping = {
        (0, 0, 0): [0,255,0],
        (1, 0, 0): [255,255,0],
        (1, 1, 0): [255,128,0],
        (1, 1, 1): [255,0,0],
    }
    return color_mapping.get(tuple(target_array), 'black')

def get_edge_mask(mask):
    img_data = np.asarray(mask[:, :], dtype=np.double)
    gx, gy = np.gradient(img_data)
    temp_edge = gy * gy + gx * gx
    temp_edge[temp_edge != 0.0] = 255.0
    temp_edge = np.asarray(temp_edge, dtype=np.uint8)
    return temp_edge

def visualize_mask(fig, axes, img, mask, score):
    # FIRST IMG: Image only -------------------------------------------------------
    # Display the original image
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis('off')  # Hide axes

    # SECOND IMG: Mask only ----------------------------------------------------
    # Display the segmentation mask
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title("Segmentation Mask")
    axes[1].axis('off')  # Hide axes

    # THIRD IMG: Mask + Image -----------------------------------------------------
    # (0, 0, 0): 'lightgreen',
    # (1, 0, 0): 'gold',
    # (1, 1, 0): 'orange',
    # (1, 1, 1): 'red',
    # Get mask's edge
    mask_edge = get_edge_mask(mask)

    # Create a composite image by overlaying the mask onto the corresponding part of the original image
    composite_image = np.copy(img)
    # Overlay in red only in regions of the mask
    #composite_image[mask != -1] = [255, 0, 0]
    composite_image[mask_edge != 0] = get_RGB_mask_color(score)

    # Display the composite image
    axes[2].imshow(composite_image)
    axes[2].set_title("Image with Overlayed Mask")
    axes[2].axis('off')  # Hide axes

    # Set the spacing between the images
    #plt.subplots_adjust(wspace=0.1)  # Adjust this value as per your needs

    return fig, axes

# Function to display and print frame information
def print_frame(frame_idx, dataset_idx, dataset):
    if dataset_idx < 0 or dataset_idx >= len(dataset):
        print(f"Error: Invalid dataset index {dataset_idx}. Dataset index must be between 0 and {len(dataset) - 1}.")
        dataset_idx = len(dataset) - 1

    video_data, target_data, mask_data = dataset[dataset_idx]
    num_frames = video_data.get_num_frames()

    if frame_idx < 0 or frame_idx >= num_frames:
        print(f"Error: Invalid frame index {frame_idx}. Frame index must be between 0 and {num_frames - 1}.")
        frame_idx = num_frames - 1

    frame = video_data.get_frame(frame_idx)
    score = target_data.get_score(frame_idx)
    mask  = mask_data.get_frame_mask(frame_idx)
    video_id = video_data.get_video_id()
    patient = video_data.get_patient()
    medical_center = video_data.get_medical_center()
    probe = video_data.get_probe_type()
    original_file = video_data.get_original_file()

    # Create an image from RGB
    img = np.zeros_like(frame, dtype=np.uint8)
    img[:, :, 0] = frame[:, :, 0]  # Red channel
    img[:, :, 1] = frame[:, :, 1]  # Green channel
    img[:, :, 2] = frame[:, :, 2]  # Blue channel

    # Create a figure with three axes
    fig, axes = plt.subplots(1, 3, figsize=(17, 9))

    fig, axes = visualize_mask(fig, axes, img, mask, score)

    # Add annotations to the image
    axes[0].annotate(f"Frame {frame_idx} \nDataset {dataset_idx}", (20, 20), color='white', fontsize=11, ha='left', va='top')

    axes[0].annotate(f"Original file: {original_file}", (20, img.shape[0] - 20), color='white', fontsize=9, ha='left', va='bottom')
    axes[0].annotate(f"Score: {score}", (20, img.shape[0] - 60), color='white', fontsize=9, ha='left', va='bottom')
    axes[0].annotate(f"Frame shape: {frame.shape}", (20, img.shape[0] - 100), color='white', fontsize=9, ha='left', va='bottom')
    axes[0].annotate(f"Probe: {probe}", (20, img.shape[0] - 140), color='white', fontsize=9, ha='left', va='bottom')
    axes[0].annotate(f"Patient: {patient}", (20, img.shape[0] - 180), color='white', fontsize=9, ha='left', va='bottom')
    axes[0].annotate(f"Medical center: {medical_center}", (20, img.shape[0] - 220), color='white', fontsize=9, ha='left', va='bottom')

    plt.show()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Function For Printing Dataloader~~~~~~~~~~~~~~~~~~~~~~~~~~
# Function to create a flexible grid for displaying frames
def create_flexible_grid(batch_size):
    num_rows = 1
    num_cols = 1
    while num_rows * num_cols < batch_size:
        if num_rows == num_cols:
            num_cols *= 2
        else:
            num_rows *= 2

    return num_rows, num_cols

def get_label_color(target_array):
    color_mapping = {
        (0, 0, 0): 'green',
        (1, 0, 0): 'gold',
        (1, 1, 0): 'orange',
        (1, 1, 1): 'red',
    }
    return color_mapping.get(tuple(target_array), 'black')

def dataloader_display_iterator(dataloader, batch_size):
    # Loop through the dataloader to load frames and targets in batches
    for (frames_batch, masks_batch ,targets_batch) in dataloader:
        if frames_batch is not None and targets_batch is not None and masks_batch is not None:
            # Get the flexible grid size for the batch
            rows, cols = create_flexible_grid(batch_size)

            # Create a subplot grid for frames
            fig, axs = plt.subplots(rows, cols, figsize=(cols, rows))
            plt.subplots_adjust(wspace=0.1, hspace=0.5)

            for i, (frame_tensor, mask_tensor, target_tensor) in enumerate(zip(frames_batch, masks_batch ,targets_batch)):
                frame_array = frame_tensor.numpy()
                mask_array = mask_tensor.numpy()
                target_array = target_tensor.numpy()

                ax = axs[i // cols, i % cols]
                ax.imshow(frame_array)
                ax.imshow(mask_array, cmap='jet', alpha=0.2) # interpolation='none'
                ax.axis('off')

                label = f"Target: {target_array}"
                color = get_label_color(target_array)
                ax.set_title(label, fontsize=7, ha='center', va='center', color=color)

                #ax.text(0.5, 0.05, label, fontsize=6.5, ha='center', va='bottom', color=color, transform=ax.transAxes)

            plt.show()
            print("\n")
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 