from torch.utils.data import Dataset
import torch
import os
import numpy as np
import cv2


class RealDataset(Dataset):
    def __init__(self, data_path, size=(128, 128)):
        super(RealDataset, self).__init__()
        self.data_path = data_path
        self.size = size
        self.filenames = os.listdir(self.data_path)[:2000]

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.data_path, self.filenames[idx]))
        img = cv2.resize(img, self.size)
        img = np.moveaxis(img, (0, 1, 2), (1, 2, 0)).astype('float32')
        return torch.Tensor(img / 255.)

    def __len__(self):
        return len(self.filenames)