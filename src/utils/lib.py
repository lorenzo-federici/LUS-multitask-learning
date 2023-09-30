# ---------------------------------------------------------------------------- #
#                                 "lib" module
#
# Library Name: lib
# Author: Lorenzo Federici
# Creation Date: September 21, 2023
# Description: This library contains a set of useful tools
# Project Name: LUS-MULTITASK-LEARNING
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
#                                    Imports                                   #
# ---------------------------------------------------------------------------- #

# import pickle
# import numpy as np
# import os
# import cv2
# import copy
# import glob
# import json
# import random
# import plots

# import tensorflow as tf
# from sklearn.metrics import accuracy_score
# from tensorflow.keras.optimizers import *
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger
# from tensorflow.keras import backend as k
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras import backend as K
# from tensorflow.keras.callbacks import *

# from itertools import groupby
# from itertools import chain

import os

import PIL
from PIL import Image
import h5py

import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uuid

from tqdm import tqdm

from pathlib import Path
from pathlib import PosixPath

import random
import pickle
from collections import defaultdict
from torch.utils.data import Subset


# ---------------------------------------------------------------------------- #
#                                    Methods                                   #
# ---------------------------------------------------------------------------- #


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ General Function ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_video_list(folder_names, base_path):
    video_list = []
    for folder_name in folder_names:
        root_folder = Path(os.path.join(base_path, folder_name))
        files = [f for f in root_folder.rglob('*.images')]
        video_list.extend(files)
    return video_list
    
def get_path_from_xlsx(xlsx_file_path, base_path, mode, folder_name = None):
    # Read xlsx file
    df = pd.read_excel(xlsx_file_path)
    df = df.dropna()

    # Create a dictionary where the key is the video's name and the value is a dictionary with the infos
    videos_dict = {}
    paths = []
    for _, row in df.iterrows():
        if mode == "iclus":
            center, patient, file_name, _, _, _, _, n_frame  = row
            if patient == "Paziente 12" and center == "No Covid Data" and file_name == "convex_movie_20":
                file_name = "convex_movie_19"
        else:
            video_id = row[0]
            center = row[1]
            patient = row[2]
            file_name = row[3]

            # Fixing missing Path
            dir_extension = ""
            if patient == "Paziente 6" and center == "Brescia":
                dir_extension = "10-3/" if file_name == "convex_202003101232160105ABD" else "16-3/"
            if  file_name in ["convex_202003131035140330ABD", "convex_202003181350200115ABD"]:
                continue # miss video dalate it

            path = f"{center}/{patient}/{dir_extension}{file_name}.images"
            paths.append(Path(f"{base_path}/Covid19/{path}"))
            
        video_info = {'medical_center': center, 'patient': patient}
        videos_dict[file_name] = video_info

    images_paths = get_video_list(folder_name, base_path) if mode == "iclus" else paths
    return images_paths, videos_dict

# Define a custom sorting function
def number_before_json(path):
    # Extract the file name
    file_name = path.name
    # Extract the number as a string between "frame_" and ".json"
    start_index = file_name.find("frame_") + len("frame_")
    end_index = file_name.find(".json")
    number_str = file_name[start_index:end_index]
    # Convert the string to an integer and use it for sorting
    return int(number_str)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     

def splitting_strategy(dataset, rseed, pkl_file, train_ratio=0.7):
    # iteration seed
    random.seed(rseed)

    # Check if the pickle file exists
    if os.path.exists(pkl_file):
        # If the pickle file exists, load the data from it
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
            medical_center_patients = data['medical_center_patients']
            data_index = data['data_index']
    else:
        # If the pickle file doesn't exist, create the data
        medical_center_patients = defaultdict(set)
        data_index = {}
        for index, (_, _, _, _, patient, medical_center) in enumerate(dataset):
            medical_center_patients[medical_center].add(patient)
            data_index[index] = (patient, medical_center)

        # Save the data to a pickle file
        data = {
            'medical_center_patients': medical_center_patients,
            'data_index': data_index
        }

        with open(pkl_file, 'wb') as f:
            pickle.dump(data, f)

    # Split the patients for each medical center
    train_indices = []
    test_indices = []

    # Lists to store statistics about medical centers and patients
    train_patients_by_center = defaultdict(set)
    test_patients_by_center = defaultdict(set)
    frame_counts_by_center = defaultdict(int)
    frame_counts_by_center_patient = defaultdict(lambda: defaultdict(int))

    for medical_center, patients in medical_center_patients.items():
        patients = list(patients)
        random.shuffle(patients)
        split_index = int(train_ratio * len(patients))

        for index, (patient, center) in data_index.items():
            if center == medical_center:
                if patient in patients[:split_index]:
                    train_indices.append(index)
                    train_patients_by_center[medical_center].add(patient)
                else:
                    test_indices.append(index)
                    test_patients_by_center[medical_center].add(patient)

                frame_counts_by_center[medical_center] += 1
                frame_counts_by_center_patient[medical_center][patient] += 1

    # Create training and test subsets
    train_dataset_subset = Subset(dataset, train_indices)
    test_dataset_subset = Subset(dataset, test_indices)

    # Sum up statistics info
    split_info = {
        'medical_center_patients': medical_center_patients,
        'frame_counts_by_center': frame_counts_by_center,
        'train_patients_by_center': train_patients_by_center,
        'test_patients_by_center': test_patients_by_center,
        'frame_counts_by_center_patient': frame_counts_by_center_patient,
        'total_train_frames': len(train_indices),
        'total_test_frames': len(test_indices)
    }

    return train_dataset_subset, test_dataset_subset, split_info  