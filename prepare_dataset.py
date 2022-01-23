import torch
import os
import torchvision
from torchvision.datasets import ImageFolder

def create_dataset(name):
    if name=='cifar10':
        data_dir = './data/cifar10'

        train_dir = data_dir + '/train'
        test_dir = data_dir + '/test'
        classes = os.listdir(train_dir)
        train_data = ImageFolder(root=train_dir,transform=torchvision.transforms.ToTensor())
        test_data = ImageFolder(root=test_dir,transform=torchvision.transforms.ToTensor())

        return train_data,test_data,classes,len(train_data)
