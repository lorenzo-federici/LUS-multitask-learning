# ---------------------------------------------------------------------------- #
#                                 "lib" module
#
# Library Name: viewlib
# Author: Lorenzo Federici
# Creation Date: September 21, 2023
# Description: This library contains a set of useful tools for frame visualization
# Project Name: LUS-MULTITASK-LEARNING
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
#                                    Imports                                   #
# ---------------------------------------------------------------------------- #
import sys
base_path = '/content/drive/MyDrive/Tesi/dataset/lollo'
if base_path + "/utils" not in sys.path:
    sys.path.append(base_path + "/utils")

import numpy as np
import matplotlib.pyplot as plt

import lib

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Function For Printing Frame~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_RGB_mask_color(target_array):
    color_mapping = {
        0 : [0,255,0],
        1 : [255,255,0],
        2 : [255,128,0],
        3 : [255,0,0],
    }
    return color_mapping.get(target_array, 'black')

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
def display_frame_data(mat_dataset, video_idx, frame_idx):
    mat_dataset_len = len(mat_dataset)
    if video_idx < 0 or video_idx >= mat_dataset_len :
        print(f"Error: Invalid video index {video_idx}. Video index must be between 0 and {mat_dataset_len - 1}.")
        video_idx = mat_dataset_len - 1

    video_target_data = mat_dataset[video_idx]
    num_frames = video_target_data.get_num_frames()

    if num_frames > 0 :
        if frame_idx < 0 or frame_idx >= num_frames :
            print(f"Error: Invalid frame index {frame_idx}. Frame index must be between 0 and {num_frames - 1}.")
            frame_idx = num_frames - 1

        # Gathering data (frames and scores)...
        frame, mask, score = video_target_data.get_data(frame_idx)

        # ...and metadata
        patient = video_target_data.get_patient()
        medical_center = video_target_data.get_medical_center()
        probe = video_target_data.get_probe_type()
        file_name = video_target_data.get_file_name()

        # Create an image from RGB
        img = np.zeros_like(frame, dtype=np.uint8)
        img[:, :, 0] = frame[:, :, 0]  # Red channel
        img[:, :, 1] = frame[:, :, 1]  # Green channel
        img[:, :, 2] = frame[:, :, 2]  # Blue channel

        fig, axes = plt.subplots(1, 3, figsize=(17, 9))
        fig, axes = visualize_mask(fig, axes, img, mask, score)

        # Add annotations to the image
        axes[0].annotate(f"Frame: {frame_idx} \nVideo: {video_idx}", (20, 20), color='white', fontsize=11, ha='left', va='top')

        axes[0].annotate(f"Original file: {file_name}", (20, img.shape[0] - 20), color='white', fontsize=9, ha='left', va='bottom')
        axes[0].annotate(f"Score: {score}", (20, img.shape[0] - 60), color='white', fontsize=9, ha='left', va='bottom')
        axes[0].annotate(f"Frame shape: {frame.shape}", (20, img.shape[0] - 100), color='white', fontsize=9, ha='left', va='bottom')
        axes[0].annotate(f"Probe: {probe}", (20, img.shape[0] - 140), color='white', fontsize=9, ha='left', va='bottom')
        axes[0].annotate(f"Patient: {patient}", (20, img.shape[0] - 180), color='white', fontsize=9, ha='left', va='bottom')
        axes[0].annotate(f"Medical center: {medical_center}", (20, img.shape[0] - 220), color='white', fontsize=9, ha='left', va='bottom')

        plt.show()
    else :
        print(f"Warning: {medical_center}/{patient}/{file_name} it's an invalid video and so not usable.")
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

# Utility function to assign color to the label based on the frame score
def get_label_color(score):
    return {0: 'green', 1: 'gold', 2: 'orange', 3: 'red'}.get(score, 'black')

def dataloader_display_iterator(dataloader):
    # TEST ->
    frames = []
    iter = 0
    # <- TEST

    # Loop through the dataloader to load frames and targets in batches
    for (frames_batch, masks_batch, targets_batch) in dataloader:
        if frames_batch is not None and targets_batch is not None and masks_batch is not None:
            # Get the actual batch size
            actual_batch_size = frames_batch.shape[0]

            # Get the flexible grid size for the batch
            rows, cols = create_flexible_grid(actual_batch_size)
            # rows, cols = create_flexible_grid(batch_size)

            # Create a subplot grid for frames
            #fig, axs = plt.subplots(rows, cols, figsize=(cols, rows))
            fig, axs = plt.subplots(rows, cols, figsize=(6, 6))
            plt.subplots_adjust(wspace=0.1, hspace=0.25)

            for i, (frame_tensor, mask_tensor, score) in enumerate(zip(frames_batch, masks_batch ,targets_batch)):
                frame_array = frame_tensor.permute(1, 2, 0).numpy()
                mask_array  = mask_tensor.permute(1, 2, 0).numpy()

                mask = lib.get_original_mask(mask_array)

                # TEST ->
                frames.append(frame_array)
                # <- TEST

                ax = axs[i // cols, i % cols]
                ax.imshow(frame_array)
                ax.imshow(mask, cmap='jet', alpha=0.2) # interpolation='none'
                ax.axis('off')

                label = f"Score: {score}"
                color = get_label_color(int(score))
                ax.set_title(label, fontsize=10, ha='center', va='top', pad=8, color=color)

            plt.show()
            print("\n")

            # TEST ->
            #break
            iter += 1

            if(iter == 5) :
                break
            # <- TEST

    # TEST ->
    print("Media:", np.mean(frames, axis=(0, 1, 2)) )
    print("Deviazione Standard:", np.std(frames, axis=(0, 1, 2)))
    # <- TEST
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 