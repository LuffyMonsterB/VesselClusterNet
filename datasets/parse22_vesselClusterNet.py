from monai import transforms
import os
from glob import glob
from torch.utils.data import Dataset, DataLoader
import torch
from einops import rearrange


class Parse22NiiDataset(Dataset):
    def __init__(self, nii_files, label_files, trans):
        super().__init__()
        self.nii_files = nii_files
        self.label_files = label_files
        self.trans = trans

    def __len__(self):
        return len(self.nii_files)

    def normalize(self, data):
        hu_max = 512
        hu_min = -968
        data[data > hu_max] = hu_max
        data[data < hu_min] = hu_min
        data = (data - (hu_min)) / (hu_max - hu_min)
        data = torch.FloatTensor(data)
        return data

    def __getitem__(self, item):
        nii_file = self.nii_files[item]
        label_file = self.label_files[item]
        file = {'image': nii_file, 'label': label_file}
        file = self.trans(file)
        nii = file['image']
        nii = rearrange(nii, "C W H D -> C D H W")
        label = file['label']
        label = rearrange(label, "C W H D -> C D H W")
        return nii, label, nii_file


def load_parse22_data(data_dir):
    files_dir = sorted(glob(os.path.join(data_dir, "*")))
    niis = []
    labels = []
    for file_dir in files_dir:
        nii_dir = os.path.join(file_dir, "image")
        label_dir = os.path.join(file_dir, "label")
        niis.append(glob(os.path.join(nii_dir, "*.nii.gz")))
        labels.append(glob(os.path.join(label_dir, "*.nii.gz")))
    return niis, labels


def get_train_dataloader(data_dir, batch_size, num_workers):
    train_niis, train_labels = load_parse22_data(data_dir)
    train_trans = transforms.Compose([
        transforms.LoadImageD(keys=['image', 'label']),
        transforms.EnsureChannelFirstD(keys=['image', 'label']),
        transforms.RandSpatialCropD(keys=['image', 'label'], roi_size=(128, 128, 64), random_size=False),
        transforms.ScaleIntensityRangeD(keys=['image'], a_min=-968, a_max=512, b_min=0, b_max=1),
        transforms.EnsureTypeD(keys=['image', 'label'])
    ])

    train_ds = Parse22NiiDataset(train_niis, train_labels, train_trans)

    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                          pin_memory=torch.cuda.is_available())
    return train_dl


def get_val_dataloader(val_dir, val_batch_size, val_numworkers):
    val_niis, val_labels = load_parse22_data(val_dir)
    val_trans = transforms.Compose([
        transforms.LoadImageD(keys=['image', 'label']),
        transforms.EnsureChannelFirstD(keys=['image', 'label']),
        transforms.ScaleIntensityRangeD(keys=['image'], a_min=-968, a_max=512, b_min=0, b_max=1),
        transforms.EnsureTypeD(keys=['image', 'label'])
    ])

    val_ds = Parse22NiiDataset(val_niis, val_labels, val_trans)

    val_dl = DataLoader(val_ds, batch_size=val_batch_size, num_workers=val_numworkers, shuffle=True,
                        pin_memory=torch.cuda.is_available())

    return val_dl
