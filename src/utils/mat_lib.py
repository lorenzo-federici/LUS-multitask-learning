# ---------------------------------------------------------------------------- #
#                                 "lib" module
#
# Library Name: lib
# Author: Lorenzo Federici
# Creation Date: September 21, 2023
# Description: This library contains a set of useful tools for .mat manipulation
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
import h5py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uuid

from tqdm import tqdm

from pathlib import Path
from pathlib import PosixPath


# ---------------------------------------------------------------------------- #
#                                    Methods                                   #
# ---------------------------------------------------------------------------- #
    
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Function For .mat Analysis~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_frames_resolutions(dataset, num_videos=None):
    video_resolutions = {}
    video_patients = {}
    broken_videos = []
    nframes = 0
    nframes_linear = 0
    nframes_convex = 0
    nframes_unclassified = 0
    total_videos = len(dataset)
    num_videos = num_videos if num_videos is not None else total_videos

    with tqdm(total=num_videos, desc="Getting frame resolutions", unit='video', dynamic_ncols=True) as pbar:
        for video_data, _, mask_data in dataset:
            resolutions = set()
            num_frames = video_data.get_num_frames()
            video_id = video_data.get_video_id()
            patient = video_data.get_patient()
            medical_center = video_data.get_medical_center()

            if num_frames == -1:
                print(f"\n>> VIDEO ERROR: Video '{video_id}' has an invalid number of frames (-1). Skipping this video.")
                broken_videos.append({
                    'video_name': video_id,
                    'patient': patient,
                    'center': medical_center,
                    'error' : 'invalid number of frames'
                })
                pbar.update(1)  # Increase the count even for broken videos
                continue
            elif num_frames != mask_data.get_num_frames():
                print(f"\n>> VIDEO ERROR: Video '{video_id}' has a different number of frames than its mask. Skipping this video.")
                broken_videos.append({
                    'video_name': video_id,
                    'patient': patient,
                    'center': medical_center,
                    'error' : 'nframe != mask frame'
                })
                pbar.update(1)  # Increase the count even for broken videos
                continue

            nframes += num_frames
            pbar.set_description(f"Getting frame resolutions (frames: {nframes})")

            probe = video_data.get_probe_type()

            if probe == 'linear':
                nframes_linear += num_frames
            elif probe == 'convex':
                nframes_convex += num_frames
            else:
                nframes_unclassified += num_frames

            for i in range(num_frames):
                frame_data = video_data.get_frame(i)
                resolution = frame_data.shape[:2]
                resolutions.add(resolution)

            video_resolutions[video_id] = resolutions
            video_patients[video_id] = (patient, medical_center, num_frames)
            pbar.update(1)

            if pbar.n == num_videos:
                break

    return video_resolutions, video_patients, nframes, nframes_linear, nframes_convex, nframes_unclassified, broken_videos
    
# Function for printing dataset information
def check_info(data_dict, dataset):
    # Access the data as needed
    pckl_resolutions = data_dict['resolutions']
    pckl_patients = data_dict['patients']
    pckl_nframes = data_dict['nframes']
    pckl_nframes_linear = data_dict['nframes_linear']
    pckl_nframes_convex = data_dict['nframes_convex']
    pckl_nframes_unclassified = data_dict['nframes_unclassified']

    # Create an empty set to track encountered video names
    encountered_video_names = set()

    # Verify if frames have consistent resolutions within each video group
    varying_resolutions_count = 0
    for video_name, resolutions_set in pckl_resolutions.items():
        if len(resolutions_set) == 1:
            print(f"Video '{video_name}' has consistent resolution: {resolutions_set.pop()}")
        else:
            varying_resolutions_count += 1
            print(f"Video '{video_name}' has varying resolutions: {resolutions_set}")

    # Verify if patients have videos with consistent resolutions within each center of research
    patient_resolutions = {}
    varying_patient_resolutions_count = 0
    for video_name, (patient, center, _) in pckl_patients.items():
        resolutions_set = pckl_resolutions[video_name]
        patient_id = f"{patient}_{center}"
        if patient_id not in patient_resolutions:
            patient_resolutions[patient_id] = resolutions_set
        else:
            if patient_resolutions[patient_id] != resolutions_set:
                varying_patient_resolutions_count += 1
                print(f"Patient '{patient}' at '{center}' has varying resolutions across videos.")

    # Print the total number of frames, frames with linear, convex, and unclassified probes, and broken videos at the end of the execution
    print(f"\nVideos: {len(dataset)}")
    # Print the count of videos with varying resolutions and varying patient resolutions
    print(f"Videos with varying resolutions: {varying_resolutions_count}")
    print(f"Patients with varying resolutions across videos: {varying_patient_resolutions_count}")
    print(f"\nFrames with convex probes: {pckl_nframes_convex} ({round(pckl_nframes_convex*100/pckl_nframes)}%)")
    print(f"Frames with linear probes: {pckl_nframes_linear} ({round(pckl_nframes_linear*100/pckl_nframes)}%)")
    print(f"Frames unclassified: {pckl_nframes_unclassified}")
    print(f"Total frames in the dataset: {pckl_nframes}")
    
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Function For .mat - .h5 conversion~~~~~~~~~~~~~~~~~~~~~~~~
# Function to save a single video data to the HDF5 file
def save_video_data(h5file, group_name, video_data, target_data, mask_data, start_index):
    # get video infos
    num_frames = video_data.get_num_frames()
    video_id = str(video_data.get_video_id())
    patient = video_data.get_patient()
    medical_center = video_data.get_medical_center()
    original_file = video_data.get_original_file()

    group = h5file.require_group(group_name)
    video_group = group.require_group(video_id)

    frames_group = video_group.require_group('frames')
    targets_group = video_group.require_group('targets')
    masks_group = video_group.require_group('masks')

    # Add patient and medical_center attributes to the video_group
    video_group.attrs['patient'] = patient
    video_group.attrs['medical_center'] = medical_center
    video_group.attrs['original_file'] = original_file

    for i in range(num_frames):
        frame_data = video_data.get_frame(i)
        frames_group.create_dataset(f'frame_{start_index + i}', data=frame_data, compression='gzip')

        target_data_i = target_data.get_score(i)
        targets_group.create_dataset(f'target_{start_index + i}', data=target_data_i, compression='gzip')

        mask_data_i = mask_data.get_frame_mask(i)
        masks_group.create_dataset(f'mask_{start_index + i}', data=mask_data_i, compression='gzip')

    # Update the 'idx_start' and 'idx_end' attributes
    video_group.attrs['frame_idx_start'] = start_index
    video_group.attrs['frame_idx_end'] = start_index + num_frames - 1

    return start_index + num_frames

# Create the HDF5 file and save the dataset
def convert_dataset_to_h5(dataset, output_file, num_videos=None):
    # Check the existence of the HDF5 file and delete it if it exists
    if os.path.exists(output_file):
        os.remove(output_file)

    with h5py.File(output_file, 'w') as h5file:
        convex_group = h5file.create_group('convex')
        linear_group = h5file.create_group('linear')

        current_index = 0

        dataset_subset = islice(dataset, num_videos) if num_videos is not None else dataset

        with tqdm(total=num_videos if num_videos is not None else len(dataset), desc="Converting dataset to HDF5", dynamic_ncols=True, unit="video") as pbar_outer:
            for video_data, target_data, mask_data in dataset_subset:
                probe = video_data.get_probe_type()

                current_index = save_video_data(h5file, probe, video_data, target_data, mask_data, current_index)

                pbar_outer.update(1)

                # Monitor file size
                file_size_gb = os.path.getsize(output_file) / (1024.0 ** 3)  # Convert to GB

                # Creating a dictionary with all the values to display in the progress bar
                postfix_dict = {
                    "file": f"{file_size_gb:.2f} GB",
                    "frames": f"{current_index}"
                }

                # Setting values in the progress bar using the dictionary
                pbar_outer.set_postfix(**postfix_dict)
                
# Function to print diagnostic information about the dataset
def print_dataset_hierarchy(h5file):
    for group_name in h5file:
        group = h5file[group_name]
        print(f"Group: {group_name}")

        num_frames_counter = 0

        for video_name in group:
            video_group = group[video_name]
            print(f"  Video: {video_name}")

            frames_group = video_group['frames']
            num_frames = len(frames_group)
            print(f"    Number of frames: {num_frames}")

            targets_group = video_group['targets']
            num_targets = len(targets_group)
            print(f"    Number of targets: {num_targets}")

            masks_group = video_group['masks']
            num_masks = len(masks_group)
            print(f"    Number of masks: {num_masks}")

            # Get patient_reference and medical_center attributes
            patient_reference = video_group.attrs['patient']
            medical_center = video_group.attrs['medical_center']
            original_file = video_group.attrs['original_file']
            print(f"    Patient Reference: {patient_reference}")
            print(f"    Medical Center: {medical_center}")
            print(f"    Original File: {original_file}")

            # Get idx attributes
            fis = video_group.attrs['frame_idx_start']
            fie = video_group.attrs['frame_idx_end']
            print(f"    Frame idx Range: {fis} - {fie}")

            continue

            print("    Frame and Target information:")
            for i in range(num_frames):
                frame_data = frames_group[f'frame_{num_frames_counter+i}']
                target_data = targets_group[f'target_{num_frames_counter+i}']
                mask_data = masks_group[f'mask_{num_frames_counter+i}']
                print(f"      Frame {num_frames_counter+i}: Shape = {frame_data.shape}, Target = {target_data.shape}, Mask = {mask_data.shape}")

            num_frames_counter += num_frames
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

