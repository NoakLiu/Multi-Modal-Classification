import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from skimage import io
import re
from io import StringIO
from sklearn.preprocessing import MultiLabelBinarizer


class TextImageDataLoader:
    def __init__(self, filename_train, filename_test, picture_path):
        self.filename_train = filename_train
        self.filename_test = filename_test
        self.picture_path = picture_path

    def load_data(self):
        df_train = self.read_csv(self.filename_train)
        df_test = self.read_csv(self.filename_test)

        return df_train, df_test

    def read_csv(self, filename):
        with open(filename) as file:
            lines = [re.sub(r'([^,])"(\s*[^\n])', r'\1/"\2', line) for line in file]
            lines = [re.sub(r'\/"\n', '"\n', line) for line in lines]
            df = pd.read_csv(StringIO(''.join(lines)), escapechar="/")
        return df

    def train_val_loader(self, dataset, val_ratio, batch_size):
        total_size = len(dataset)
        train_size = int(total_size * (1 - val_ratio))
        val_size = total_size - train_size

        train_set, val_set = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    def test_loader(self, test_dataset, batch_size):
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return test_loader


class TextImageDataset(Dataset):
    def __init__(self, image_file, text_file, target=None, img_transform=None, picture_path=''):
        self.text_file = text_file
        self.image_file = image_file
        self.target = target
        self.img_transform = img_transform
        self.picture_path = picture_path

    def __len__(self):
        return len(self.image_file)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.image_file[idx]
        image_path = self.picture_path + str(img_name)
        img = Image.open(image_path)

        if self.img_transform:
            img = self.img_transform(img)

        text = torch.Tensor(self.text_file[idx]).to(torch.int64)
        target = torch.tensor(self.target[idx]) if self.target is not None else None

        sample = {'image': img, 'text': text, 'target': target}
        return sample
