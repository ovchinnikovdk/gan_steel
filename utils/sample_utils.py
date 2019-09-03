from torch.utils.data import Dataset
import torch
import os
import cv2


class RandomNoiseDataset(Dataset):
    def __init__(self, lin_size=256, dataset_size=2000):
        super(RandomNoiseDataset, self).__init__()
        self.lin_size = lin_size
        self.dataset_size = dataset_size

    def __getitem__(self, index: int):
        return torch.randn(self.lin_size)

    def __len__(self):
        return self.dataset_size


class RealDataset(Dataset):
    def __init__(self, data_path, size=(128, 128)):
        super(RealDataset, self).__init__()
        self.data_path = data_path
        self.size = size
        self.filenames = os.listdir(self.data_path)

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.data_path, self.filenames[idx]))
        img = cv2.resize(img, self.size)
        return img

    def __len__(self):
        return len(self.filenames)
