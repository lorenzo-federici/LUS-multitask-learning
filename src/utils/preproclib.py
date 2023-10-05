# ---------------------------------------------------------------------------- #
#                                 "lib" module
#
# Library Name: preproclib
# Author: Lorenzo Federici
# Creation Date: September 21, 2023
# Description: This library contains a set of useful tools for mask manipulation
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
from itertools import islice
import numpy as np
import pandas as pd
import uuid
from tqdm import tqdm
from pathlib import Path

# ---------------------------------------------------------------------------- #
#                                    Methods                                   #
# ---------------------------------------------------------------------------- #

# Medical center and patients bind for 'No Covid Data'
ncdpatient_to_medcenter_dict = {
    "Paziente 1": "Lucca (NCD)",
    "Paziente 2": "Lucca (NCD)",
    "Paziente 3": "Lucca (NCD)",
    "Paziente 4": "Lucca (NCD)",
    "Paziente 5": "Lucca (NCD)",
    "Paziente 6": "Lucca (NCD)",
    "Paziente 7": "Lucca (NCD)",
    "Paziente 8": "Tione (NCD)",
    "Paziente 9": "Gemelli - Roma (NCD)",
    "Paziente 10": "Tione (NCD)",
    "Paziente 11": "Gemelli - Roma (NCD)",
    "Paziente 12": "Gemelli - Roma (NCD)",
    "Paziente 13": "Gemelli - Roma (NCD)",
    "Paziente 14": "Gemelli - Roma (NCD)",
}

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Function For Mask~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def img_data_to_pil(img_data):
    f = io.BytesIO()
    f.write(img_data)
    img_pil = Image.open(f)
    return img_pil

def img_data_to_arr(img_data):
    img_pil = img_data_to_pil(img_data)
    img_arr = np.array(img_pil)
    return img_arr

def img_b64_to_arr(img_b64):
    img_data = base64.b64decode(img_b64)
    img_arr = img_data_to_arr(img_data)
    return img_arr

def convert_json_to_mask(json_file):
    data = json.load(open(json_file))

    imageData = data.get('imageData')

    # se per qualche motivo il json è rotto allora prendo li png nella cartella
    if not imageData:
        imagePath = os.path.join(os.path.dirname(json_file), data['imagePath'])
        with open(imagePath, 'rb') as f:
            imageData = f.read()
            imageData = base64.b64encode(imageData).decode('utf-8')
    img = img_b64_to_arr(imageData)

    label_name_to_value = {'_background_': -1, '0':0,'1':1,'2':2,'3':3}
    for shape in sorted(data['shapes'], key=lambda x: x['label']):
        label_name = shape['label']
    lbl, _ = shapes_to_label(img.shape, data['shapes'], label_name_to_value)
    return lbl

def shape_to_mask(img_shape, points, shape_type=None,line_width=10, point_size=5):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    if shape_type == 'circle':
        assert len(xy) == 2, 'Shape of shape_type=circle must have 2 points'
        (cx, cy), (px, py) = xy
        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
    elif shape_type == 'rectangle':
        assert len(xy) == 2, 'Shape of shape_type=rectangle must have 2 points'
        draw.rectangle(xy, outline=1, fill=1)
    elif shape_type == 'line':
        assert len(xy) == 2, 'Shape of shape_type=line must have 2 points'
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == 'linestrip':
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == 'point':
        assert len(xy) == 1, 'Shape of shape_type=point must have 1 points'
        cx, cy = xy[0]
        r = point_size
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
    else:
        assert len(xy) > 2, 'Polygon must have points more than 2'
        draw.polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask

def shapes_to_label(img_shape, shapes, label_name_to_value):

    cls = np.ones(img_shape[:2], dtype=np.int32)*label_name_to_value['_background_']
    ins = np.ones_like(cls)*label_name_to_value['_background_']
    instances = []
    for shape in shapes:
        points = shape['points']
        label = shape['label']
        group_id = shape.get('group_id')
        if group_id is None:
            group_id = uuid.uuid1()
        shape_type = shape.get('shape_type', None)

        cls_name = label
        instance = (cls_name, group_id)

        if instance not in instances:
            instances.append(instance)
        ins_id = instances.index(instance) + 1
        cls_id = label_name_to_value[cls_name]

        mask = shape_to_mask(img_shape[:2], points, shape_type)
        cls[mask] = cls_id
        ins[mask] = ins_id

    return cls, ins

def stringToRGB(base64_string):
    imgdata = base64.b64decode(str(base64_string))
    img = Image.open(io.BytesIO(imgdata))
    opencv_img= cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    return opencv_img

def get_mask(json_list):
    masks_list = []
    imgs_list  = []

    # TODO: gestire img e le shape delle maschere
    #with open(json_list[0], 'r') as file:
    #    data = json.load(file)

    for json_file in json_list:
        with open(json_file, 'r') as file:
            data = json.load(file)

        imageData = data.get('imageData')
        img = stringToRGB(imageData)

        label_name_to_value = {'_background_': -1, '0':0,'1':1,'2':2,'3':3}
        for shape in sorted(data['shapes'], key=lambda x: x['label']):
            label_name = shape['label']
        lbl, _ = shapes_to_label(img.shape, data['shapes'], label_name_to_value)
        masks_list.append(lbl)
        imgs_list.append(img)

    # list to mask
    masks_mat = np.stack(masks_list, axis=-1)
    imgs_mat  = np.stack(imgs_list, axis=-1)
    return [masks_mat, imgs_mat]
   
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
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
    if mode == "iclus":
        df = df.reset_index()
        df['index'] = df.index + 1
    df = df.iloc[:, :4].dropna()

    # Create a dictionary where the key is the video's name and the value is a dictionary with the infos
    videos_dict = {}
    paths = []
    for _, row in df.iterrows():
        _, center, patient, file_name = row
        if patient == "Paziente 12" and center == "No Covid Data" and file_name == "convex_movie_20":
            file_name = "convex_movie_19"

        if mode == "clinic-eval":
            # Fixing missing Path
            dir_extension = ""
            if patient == "Paziente 6" and center == "Brescia":
                dir_extension = "10-3/" if file_name == "convex_202003101232160105ABD" else "16-3/"
            if  file_name in ["convex_202003131035140330ABD", "convex_202003181350200115ABD"]:
                continue # miss video dalate it
            path = f"{center}/{patient}/{dir_extension}{file_name}.images"
            paths.append(Path(f"{base_path}/Covid19/{path}"))

        if center == "No Covid Data":
            center = ncdpatient_to_medcenter_dict.get(patient, "Unknown")

        # list of all .images folder path
        images_paths = get_video_list(folder_name, base_path) if mode == "iclus" else paths

        video_info = {'medical_center': center, 'patient': patient}
        videos_dict[file_name] = video_info


    return images_paths, videos_dict

def frames_res_study(dataset, num_videos=None):
    video_resolutions = {}
    video_patients    = {}
    video_frames      = []
    invalid_videos    = []
    nframes_total  = 0
    nframes_linear = 0
    nframes_convex = 0
    nframes_unclassified = 0
    mat_dataset_len = len(dataset)
    num_videos = num_videos if num_videos is not None else mat_dataset_len

    with tqdm(total=num_videos, desc="Gathering info from mat dataset (frames=0) ", unit='video', dynamic_ncols=True) as pbar:
        for _, video_target_data in enumerate(dataset):
            resolutions = set()
            file_name = video_target_data.get_file_name()
            num_frames = video_target_data.get_num_frames()

            video_frames.append(num_frames)

            if num_frames == -1:
                print(f"\nError: Video '{file_name}' has an invalid number of frames (-1). Skipping this video.")
                invalid_videos.append({
                    'video_name': file_name,
                    'patient': video_target_data.get_patient(),
                    'center': video_target_data.get_medical_center()
                })
                pbar.update(1)  # Increase the count even for broken videos
                continue

            nframes_total += num_frames
            pbar.set_description(f"Gathering info from mat dataset (frames={nframes_total}) ")

            patient        = video_target_data.get_patient()
            medical_center = video_target_data.get_medical_center()
            probe          = video_target_data.get_probe_type()

            if probe == 'linear':
                nframes_linear += num_frames
            elif probe == 'convex':
                nframes_convex += num_frames
            else:
                nframes_unclassified += num_frames

            for i in range(num_frames):
                frame_data = video_target_data.get_frame(i)
                resolution = frame_data.shape[:2]
                resolutions.add(resolution)

            video_resolutions[file_name] = resolutions
            video_patients[file_name] = (patient, medical_center, num_frames)
            pbar.update(1)

            if pbar.n == num_videos:
                break

    return video_resolutions, video_patients, video_frames, nframes_total, nframes_linear, nframes_convex, nframes_unclassified, invalid_videos


# Function to save a single video data to the HDF5 file
def save_video_data(h5file, video_target_data, resuming, start_index):
    # get video data and metadata
    num_frames     = video_target_data.get_num_frames()
    video_name     = str(video_target_data.get_file_name())
    patient        = video_target_data.get_patient()
    medical_center = video_target_data.get_medical_center()
    source_file     = video_target_data.get_source_file_name()

    # Get the probe (it correspond to the h5 macro-group)
    probe = video_target_data.get_probe_type()
    group = h5file.require_group(probe)

    # Create the video group (making sure to handle the known duplicates)
    group_name = video_name
    
    video_group = group.require_group(group_name)

    # Ensure the existence of groups to contain frame and score data
    frames_group  = video_group.require_group('frames')
    masks_group   = video_group.require_group('masks')
    targets_group = video_group.require_group('targets')

    # Add 'patient', 'medical_center' and 'source_file' attributes to the video_group
    video_group.attrs['patient']        = patient
    video_group.attrs['medical_center'] = medical_center
    video_group.attrs['source_file']     = source_file

    for i in range(num_frames):
        frame, mask, score = video_target_data.get_data(i)

        frame_dset_name  = f'frame_{start_index + i}'
        mask_dset_name   = f'mask_{start_index + i}'
        target_dset_name = f'target_{start_index + i}'

        if not resuming:
            frames_group.create_dataset(frame_dset_name, data=frame, compression='gzip')
            masks_group.create_dataset(mask_dset_name, data=mask, compression='gzip')
            targets_group.create_dataset(target_dset_name, data=int(score))
        else:
            if frame_dset_name not in frames_group:
                frames_group.create_dataset(frame_dset_name, data=frame, compression='gzip')
            if mask_dset_name not in masks_group:
                masks_group.create_dataset(mask_dset_name, data=mask, compression='gzip')
            if target_dset_name not in targets_group:
                targets_group.create_dataset(target_dset_name, data=int(score))

    # Update the 'idx_start' and 'idx_end' attributes
    video_group.attrs['frame_idx_start'] = start_index
    video_group.attrs['frame_idx_end'] = start_index + (num_frames - 1)

    return start_index + num_frames

# Create the HDF5 file and save the dataset
def convert_matdataset_to_h5(mat_dataset, output_file, dataset_conv_checkpoint_path, num_videos=None, checkpoint_idx_interval=10):
    # set the conversion conditions
    mat_dataset_len = len(mat_dataset)
    num_videos = num_videos if num_videos is not None else mat_dataset_len
    checkpoint_exists = os.path.exists(dataset_conv_checkpoint_path)
    h5_dataset_exists = os.path.exists(output_file)
    current_frame_index = 0
    current_video_index = 0
    resuming = False
    start = True

    # Load progress if checkpoint and dataset files exist
    if h5_dataset_exists and checkpoint_exists :
        with open(dataset_conv_checkpoint_path, 'r') as checkpoint:
            progress = checkpoint.read().split(',')
            if len(progress) == 2:
                current_frame_index = int(progress[0])
                current_video_index = int(progress[1]) + 1
                resuming = True

                print(f"Last checkpoint loaded successfully (resuming from the {current_video_index}th video).\n")
    else :
        # No progress file detected, can start a new conversion
        if h5_dataset_exists:
            if input("There's already an .h5 dataset file, do you want to overwrite it? (y/N): ").strip().lower() == 'y':
                os.remove(output_file)
            else :
                start = False

    # Starting the conversion writing the HDF5 dataset in 'output_file'
    if start :
        with h5py.File(output_file, 'a' if resuming else 'w') as h5file:
            # Slicing the dataset if 'num_videos' is provided (creating a subset)
            mat_dataset_subset = islice(mat_dataset, num_videos) if num_videos != mat_dataset_len else mat_dataset

            # if it's a new conversion create the h5 macro-groups
            if not resuming :
                convex_group = h5file.create_group('convex')
                linear_group = h5file.create_group('linear')
            else :
                # Slice the dataset to resume from the next video
                mat_dataset_subset = islice(mat_dataset_subset, current_video_index, None)

            # Iterate through the videos of the mat dataset
            with tqdm(total=num_videos, desc="Converting mat dataset to HDF5", dynamic_ncols=True, unit="video", initial=current_video_index) as pbar_outer:
                for video_index, video_target_data in enumerate(mat_dataset_subset, start=current_video_index):
                    # Save the video and target data into the h5 dataset
                    current_frame_index = save_video_data(h5file, video_target_data, resuming, current_frame_index)

                    # Update the progress bar
                    pbar_outer.update(1)
                    pbar_outer.set_postfix(file=f"{os.path.getsize(output_file) / (1024.0 ** 3):.2f} GB", frames=current_frame_index)

                    # Save the checkpoint every 'checkpoint_idx_interval' videos
                    if video_index % checkpoint_idx_interval == 0:
                        with open(dataset_conv_checkpoint_path, 'w') as checkpoint:
                            checkpoint.write(f"{current_frame_index},{video_index}")

                    # Delete checkpoint file when conversion is completed
                    if pbar_outer.n == num_videos:
                        os.remove(dataset_conv_checkpoint_path)
                        break

        # Print a message to indicate that the data has been saved
        print("\nDataset converted and saved to:", output_file)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     

