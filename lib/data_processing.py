#from torchvision import transforms
import torch
import torchvision.transforms.v2 as transforms
from pathlib import Path
import pandas as pd
import numpy as np
from lib.dataset import ImagesDataset
import config.opt as opt

DIR_MAIN = Path(opt.dataset_root)
DIR_TRAIN = DIR_MAIN / 'train'
DIR_VAL = DIR_MAIN / 'val'


def collect_data(dir):
    files = sorted(list((dir / 'images').glob('*.png')))
    labels = []
    lbl_df = pd.read_csv(dir / 'labels.txt', sep=' ', index_col=0, header=None)
    for file in files:
        lbl = (lbl_df.loc[file.name]).to_numpy().astype('float32')
        labels.append(lbl)
    return files, labels


def trainval_dataset():
    attributes = opt.attributes

    files_train, labels_train = collect_data(DIR_TRAIN)
    files_val, labels_val = collect_data(DIR_VAL)

    # transforms_train = transforms.Compose([
    #     transforms.Resize(opt.resize),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(opt.mean, opt.std),
    # ])
    #
    # transforms_val = transforms.Compose([
    #     transforms.Resize(opt.resize),
    #     transforms.ToTensor(),
    #     transforms.Normalize(opt.mean, opt.std)
    # ])

    transforms_train = transforms.Compose([
        transforms.Resize(opt.resize),
        #transforms.Pad(32, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToImageTensor(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(opt.mean, opt.std),
    ])

    transforms_val = transforms.Compose([
        transforms.Resize(opt.resize),
        #transforms.Pad(32, padding_mode='reflect'),
        transforms.ToImageTensor(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(opt.mean, opt.std)
    ])

    train_dataset = ImagesDataset(files=files_train,
                                  labels=labels_train,
                                  attributes=attributes,
                                  transforms=transforms_train,
                                  modelinput=opt.resize,
                                  mode='train')

    val_dataset = ImagesDataset(files=files_val,
                                labels=labels_val,
                                attributes=attributes,
                                transforms=transforms_val,
                                modelinput=opt.resize,
                                mode='val')

    return train_dataset, val_dataset


def test_dataset(test_dir):
    attributes = opt.attributes
    files_test = sorted(list(test_dir.glob('*.png')))

    transforms_test = transforms.Compose([
        transforms.Resize(opt.resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=opt.mean, std=opt.std)
    ])
    test_dataset = ImagesDataset(files=files_test,
                                 labels=None,
                                 attributes=attributes,
                                 transforms=transforms_test,
                                 modelinput=opt.resize,
                                 mode='test')
    return test_dataset