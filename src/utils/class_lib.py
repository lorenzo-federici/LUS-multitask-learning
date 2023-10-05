from tqdm import tqdm
import h5py
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Classe personalizzata per importare i dati dai file .h5 in un dataloader
class HDF5Dataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.h5file = h5py.File(file_path, 'r')
        self.group_names = list(self.h5file.keys())
        self.total_videos = sum(len(self.h5file[group_name]) for group_name in self.group_names)
        self.total_frames, self.frame_index_map = self.calculate_total_frames_and_index_map()

        print(f"\n{self.total_videos} videos ({self.total_frames} frames) loaded.")

    def calculate_total_frames_and_index_map(self):
        max_frame_idx_end = 0
        frame_index_map = {}

        # Create tqdm progress bar
        with tqdm(total=self.total_videos, desc="Calculating frames and index map", unit='video', dynamic_ncols=True) as pbar:
            for group_name in self.group_names:
                for video_name in self.h5file[group_name]:
                    video_group = self.h5file[group_name][video_name]
                    frame_idx_start = video_group.attrs['frame_idx_start']
                    frame_idx_end = video_group.attrs['frame_idx_end']
                    max_frame_idx_end = max(max_frame_idx_end, frame_idx_end)
                    for i in range(frame_idx_start, frame_idx_end + 1):
                        frame_index_map[i] = (group_name, video_name)
                    pbar.update(1)  # Update progress bar for each video

        total_frames = max_frame_idx_end + 1

        return total_frames, frame_index_map

    def __len__(self):
        return self.total_frames

    def __getitem__(self, index):
        if index < 0 or index >= self.total_frames:
            raise IndexError("Index out of range")

        group_name, video_name = self.frame_index_map[index]
        video_group = self.h5file[group_name][video_name]
        frame_data = video_group['frames'][f'frame_{index}'][:]
        target_data = video_group['targets'][f'target_{index}'][:]
        mask_data = video_group['masks'][f'mask_{index}'][:]

        # get metadata
        patient = video_group.attrs['patient']
        medical_center = video_group.attrs['medical_center']

        #return index, frame_tensor, target_data
        return index, frame_data, mask_data, target_data, patient, medical_center
    

# Custom replica class of the dataset to train the neural network (return -> [frame,target])
class FrameTargetDataset(Dataset):
    def __init__(self, hdf5_dataset, transform = None):
        self.hdf5_dataset = hdf5_dataset
        #self.resize_size = (100, 150)
        self.resize_size = (224, 224)
        self.transform = transform

    def __len__(self):
        return len(self.hdf5_dataset)

    def __getitem__(self, index):
        _, frame_data, mask_data, target_data, _, _ = self.hdf5_dataset[index]

        if self.transform is not None:
            frame_tensor, mask_tensor = self.transform(frame_data, mask_data)

        # # Apply Resize transformation
        # frame_tensor = transforms.ToTensor()(frame_data)
        # frame_tensor = transforms.Resize(self.resize_size, antialias=True)(frame_tensor)
        # frame_tensor = frame_tensor.permute(1, 2, 0) # Move channels to the last dimension (needed after resize)

        # # Apply Resize transformation
        # mask_tensor = transforms.ToTensor()(mask_data)
        # mask_tensor = transforms.Resize(self.resize_size, antialias=True)(mask_tensor)
        # mask_tensor = mask_tensor.permute(1, 2, 0) # Move channels to the last dimension (needed after resize)


        return frame_tensor, mask_tensor
