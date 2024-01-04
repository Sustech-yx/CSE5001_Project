import os
import numpy as np
from torch.utils.data import Dataset

class MNISTDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.classes =  [file for file in os.listdir(self.root_dir) if not file.startswith('.')]
        self.transform = transform
        self.data = self.load_data()

    def load_data(self):
        data = []
        for class_folder in self.classes:
            class_path = os.path.join(self.root_dir, class_folder)
            for file_name in os.listdir(class_path):
                file_path = os.path.join(class_path, file_name)
                data.append((file_path, int(class_folder)))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path, label = self.data[idx]
        data = np.load(file_path)
        if self.transform:
            data = self.transform(data)
        return data, label
    

class NONOISE_MNISTDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.classes =  [file for file in os.listdir(self.root_dir) if not file.startswith('.')]
        self.transform = transform
        self.data = self.load_data()

    def load_data(self):
        data = []
        for class_folder in self.classes:
            class_path = os.path.join(self.root_dir, class_folder)
            