import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset

import os
import sys
import time


def check_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class Logger:
    def __init__(self, file, print=True):
        self.file = file
        local_time = time.strftime("%b%d_%H%M%S", time.localtime()) 
        #self.file += local_time
        self.All_file = 'logs/All.log'
        
    def print(self, content='', end = '\n', file=None):
        if file is None:
            file = self.file
        with open(file, 'a') as f:
            if isinstance(content, str):
                f.write(content+end)
            else:
                old=sys.stdout 
                sys.stdout = f
                print(content)
                sys.stdout = old
        if file is None:
            self.print(content, file=self.All_file)
        print(content,end=end)


def get_test_cifar(batch_size):
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    return test_loader

def prepare_cifar(batch_size, test_batch_size):
    kwargs = {'num_workers': 8, 'pin_memory': True}
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, **kwargs)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader

class TensorDataset(Dataset):
    def __init__(self, *tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        im, targ = tuple(tensor[index] for tensor in self.tensors)
        if self.transform:
            real_transform = transforms.Compose([
                transforms.ToPILImage(),
                self.transform
            ])
            im = real_transform(im)
        return im, targ

    def __len__(self):
        return self.tensors[0].size(0)

def prepare_robust_cifar(batch_size, test_batch_size):
    data_path = "data/release_datasets/d_robust_CIFAR/"
    kwargs = {'num_workers': 8, 'pin_memory': True}
    train_data = torch.cat(torch.load(os.path.join(data_path, f"CIFAR_ims")))
    train_labels = torch.cat(torch.load(os.path.join(data_path, f"CIFAR_lab")))

    transform_train = transforms.Compose([transforms.ToTensor(),])
    transform_test = transforms.Compose([transforms.ToTensor(),])
    train_dataset = TensorDataset(train_data, train_labels, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader
