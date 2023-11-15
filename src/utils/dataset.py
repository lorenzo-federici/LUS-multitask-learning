"""Import modules."""
import os
import pickle
import random
from collections import defaultdict
import h5py
import tensorflow as tf
from keras.utils import Sequence, to_categorical
from tqdm import tqdm
import albumentations as A

class RichHDF5Dataset(Sequence):
    def __init__(self, file_path, pkl_frame_idxmap_path):
        self.file_path = file_path
        self.pkl_file_path = pkl_frame_idxmap_path
        self.h5file = h5py.File(file_path, 'r')
        self.group_names = list(self.h5file.keys())
        self.total_videos, self.total_frames, self.frame_index_map = self.elaborate_frameidx_map()

    def load_cached_data(self):
        if os.path.exists(self.pkl_file_path):
            with open(self.pkl_file_path, 'rb') as pickle_file:
                return pickle.load(pickle_file)
        else:
            return None

    def save_cached_data(self, data):
        with open(self.pkl_file_path, 'wb') as pickle_file:
            pickle.dump(data, pickle_file)

    def elaborate_frameidx_map(self):
        # Try to load cached data
        cached_data = self.load_cached_data()
        if cached_data is not None:
            total_videos, total_frames, frame_index_map = cached_data
            print(f"{total_videos} videos ({total_frames} frames) loaded from cached data.")
        else:
            total_videos = 0
            max_frame_idx_end = 0
            frame_index_map = {}

            # Create tqdm progress bar
            with tqdm(desc="Elaborating frames and index mapping", unit=' video', dynamic_ncols=True) as pbar:
                for group_name in self.group_names:
                    for video_name in self.h5file[group_name]:
                        video_group = self.h5file[group_name][video_name]
                        frame_idx_start = video_group.attrs['frame_idx_start']
                        frame_idx_end = video_group.attrs['frame_idx_end']
                        max_frame_idx_end = max(max_frame_idx_end, frame_idx_end)
                        for i in range(frame_idx_start, frame_idx_end + 1):
                            frame_index_map[i] = (group_name, video_name)
                        total_videos += 1
                        pbar.update(1)

            total_frames = max_frame_idx_end + 1

            # Save data to pickle file for future use
            self.save_cached_data((total_videos, total_frames, frame_index_map))
            print(f"\n{total_videos} videos ({total_frames} frames) loaded and cached.")

        return total_videos, total_frames, frame_index_map

    def __len__(self):
        return self.total_frames

    def __getitem__(self, index):
        if index < 0 or index >= self.total_frames:
            raise IndexError("Index out of range")

        group_name, video_name = self.frame_index_map[index]
        video_group = self.h5file[group_name][video_name]
        frame_data  = video_group['frames'][f'frame_{index}'][:]
        mask        = video_group['masks'][f'mask_{index}'][:]
        target_data = video_group['targets'][f'target_{index}']

        # Trasform mask into binary
        mask_data = mask + 1
        mask_data[mask_data > 1] = 1

        # get metadata
        patient        = video_group.attrs['patient']
        medical_center = video_group.attrs['medical_center']

        return (index, frame_data, target_data, mask_data, patient, medical_center)

class CustomAugmentation:
    def __init__(self, augmentation = False):
        self.augmentation = augmentation

        self.aug_false = A.Resize(224, 224, p=1, always_apply=True)
        self.aug_true = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.GaussianBlur(blur_limit=(1, 3), p=0.3),
            A.Rotate(limit=(-23, 23), p=0.3),
            A.GaussNoise(var_limit=0.01, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=(0.9, 1.1), p=0.3),
            A.Resize(224, 224, p=1, always_apply=True)
        ], p=0.5)

        
    def __call__(self, frame, mask):
        if self.augmentation: 
            augmented = self.aug_true(image=frame, mask=mask)
        else:
            augmented = self.aug_false(image=frame, mask=mask)
        return augmented['image'], augmented['mask']

class TFDataset():
    def __init__(self, hdf5_dataset, indexes, batch_size=16, task = 'multitask', is_train=False):
        self.hdf5_dataset = hdf5_dataset
        self.indexes      = indexes
        self.batch_size   = batch_size
        self.is_train     = is_train
        self.task         = task

        self.dataset = None

    def _process_data(self, frame_data, target_data):
        frame_tensor = tf.cast(frame_data, tf.float32) / 255.0
        
        label = tf.squeeze(target_data)
        label_ohe = tf.one_hot(label, depth=4, dtype=tf.int32)

        # if self.task == 'classification':
        #     return frame_tensor, label_ohe
        # elif self.task == 'segmentaiton':
        #     return frame_tensor, mask_data
        # elif self.task == 'multitask':
        #     return frame_tensor, label_ohe, mask_data
        return frame_tensor, label_ohe
    
    def _generator(self):
        for index in self.indexes:
            _, frame_data, target_data, mask_data, _, _ = self.hdf5_dataset[index]

            data_augmentation = CustomAugmentation(self.is_train)
            frame_data, mask_data = data_augmentation(frame_data, mask_data)

            yield frame_data, target_data

    def _map(self, index):
        index = index.numpy()
        _, frame_data, target_data, mask_data, _, _ = self.hdf5_dataset[index]
        
        data_augmentation = CustomAugmentation(self.is_train)
        frame_data, mask_data = data_augmentation(frame_data, mask_data)

        return frame_data, target_data

    def generate_data(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.indexes)
    
        if self.is_train:
            dataset = dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
        
        dataset = dataset.map(lambda index: tf.py_function(func=self._map, inp=[index], Tout=(tf.float32, tf.int32)),
                                num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(lambda f, t: tf.py_function(func=self._process_data, inp=[f, t], Tout=(tf.float32, tf.int32)),
                                num_parallel_calls=tf.data.AUTOTUNE)
        
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.repeat()
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        self.dataset = dataset

        return dataset

    def as_iterator(self):
        if self.dataset is None:
            self.generate_data()
        
        return self.dataset.as_numpy_iterator()

# ------------------------------------------------------------------------------------------------------
# Datset Function
# ------------------------------------------------------------------------------------------------------

def _load_dsdata_pickle(dataset, pkl_file):
    # Check if the pickle file exists
    if pkl_file and os.path.exists(pkl_file):
        # If the pickle file exists, load the data from it
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
            medical_center_patients = data['medical_center_patients']
            data_index = data['data_index']
            data_map_idxs_pcm = data['data_map_idxs_pcm']
            score_counts = data['score_counts']
            labels = data['labels']
    else:
        # If the pickle file doesn't exist, create the data
        medical_center_patients = defaultdict(set)
        data_index = {}
        data_map_idxs_pcm = defaultdict(list)
        score_counts = defaultdict(int)
        labels = []  # List to store target labels

        for index, (_, _, target_data, _, patient, medical_center) in enumerate(tqdm(dataset)):
        
            medical_center_patients[medical_center].add(patient)
            data_index[index] = (patient, medical_center)
            data_map_idxs_pcm[(patient, medical_center)].append(index)
            score_counts[int(target_data[()])] += 1
            labels.append(int(target_data[()]))
        
        # Save the data to a pickle file if pkl_file is provided
        if pkl_file:
            data = {
                'medical_center_patients': medical_center_patients,
                'data_index': data_index,
                'data_map_idxs_pcm': data_map_idxs_pcm,
                'score_counts': score_counts,
                'labels': labels
            }
            
            with open(pkl_file, 'wb') as f:
                pickle.dump(data, f)
    
    return medical_center_patients, data_index, data_map_idxs_pcm, score_counts, labels

def split_strategy(dataset, ratios=[0.6, 0.2, 0.2], pkl_file=None, rseed=42):
    # Set the random seed for repeatability
    random.seed(rseed)

    if len(ratios) == 2:
        train_ratio, _ = ratios
        val_ratio = 0.0
    elif len(ratios) == 3:
        train_ratio, val_ratio, _ = ratios
    else:
        raise ValueError("ratios list must have 1, 2, or 3 values that sum to 1.0")
    
    # 0. gather the metadata
    medical_center_patients, data_index, data_map_idxs_pcm, score_counts, labels = _load_dsdata_pickle(dataset, pkl_file)    
    
    # 1. calculate the target number of frames for each split
    total_frames = len(labels)
    train_frames = int(total_frames * train_ratio)
    val_frames = int(total_frames * val_ratio)
    test_frames = total_frames - train_frames - val_frames
    
    # 2. splitting the dataset by patients taking into account sets split ratio
    # sets
    train_idxs = []
    val_idxs = []
    test_idxs = []

    # sets to store statistics about medical centers and patients
    train_patients_by_center = defaultdict(set)
    val_patients_by_center = defaultdict(set)
    test_patients_by_center = defaultdict(set)

    # 2.1 test set
    while (len(test_idxs) < test_frames):
        center = random.choice(list(medical_center_patients.keys()))
        patients = medical_center_patients[center]
        try:
            patient = patients.pop()
            test_idxs.extend(data_map_idxs_pcm[(patient, center)])
            test_patients_by_center[center].add(patient)
        except:
            del medical_center_patients[center]
        
    # 2.2 validation set
    while (len(val_idxs) < val_frames):
        center = random.choice(list(medical_center_patients.keys()))
        patients = medical_center_patients[center]
        try:
            patient = patients.pop()
            val_idxs.extend(data_map_idxs_pcm[(patient, center)])
            val_patients_by_center[center].add(patient)
        except:
                del medical_center_patients[center]

    # 2.3 training set
    for center in list(medical_center_patients.keys()):
        for patient in list(medical_center_patients[center]):
            train_idxs.extend(data_map_idxs_pcm[(patient, center)])
            train_patients_by_center[center].add(patient)
    
    # 4. compliance checks
    total_frames_calc = len(train_idxs) + len(val_idxs) + len(test_idxs)
    if total_frames != total_frames_calc:
        raise ValueError(f"splitting gone wrong (expected: {total_frames}, got:{total_frames_calc})")
    
    # 5. sum up statistics info
    split_info = {
        'medical_center_patients': medical_center_patients,
        'train_patients_by_center': train_patients_by_center,
        'val_patients_by_center': val_patients_by_center,
        'test_patients_by_center': test_patients_by_center,
        'score_counts': score_counts,
        'labels': labels
    }

    if val_ratio == 0.0:
        return train_idxs, test_idxs, split_info

    return train_idxs, val_idxs, test_idxs, split_info
