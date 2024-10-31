from torch.utils.data import DataLoader, DistributedSampler, Dataset
from torchvision import datasets, transforms, models
from datasets import load_dataset
import os
import torchvision
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

class ImageNet100Dataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']

        if self.transform:
            image = self.transform(image)

        return image, label

def load_imagenet100(num_replicas, rank):
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB')),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    ds = load_dataset("clane9/imagenet-100", cache_dir="./data/imagenet-100")
    train_dataset = ImageNet100Dataset(ds['train'], transform=transform)
    val_dataset = ImageNet100Dataset(ds['validation'], transform=transform)
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=num_replicas, rank=rank)
    train_dataloader = DataLoader(train_dataset, batch_size=128, sampler=train_sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    return train_dataloader, val_dataloader

def load_cv_data(dataset, num_replicas, rank):
    if dataset=='imagenet100':
        train_dataloader, val_dataloader = load_imagenet100(num_replicas=num_replicas, rank=rank)
        return train_dataloader, val_dataloader, 100
    if dataset=='cifar10':
        train_dataloader, val_dataloader = load_cifar10(num_replicas=num_replicas, rank=rank)
        return train_dataloader, val_dataloader, 10
    if dataset=='cifar100':
        train_dataloader, val_dataloader = load_cifar100(num_replicas=num_replicas, rank=rank)
        return train_dataloader, val_dataloader, 100

def load_cifar10(num_replicas, rank):
    data_dir = './data/cifar10'
    transform = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    )

    train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=num_replicas, rank=rank)
    train_dataloader = DataLoader(train_dataset, batch_size=512, sampler=train_sampler)
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    val_loader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False)
    return train_dataloader, val_loader

def load_cifar100(num_replicas, rank):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    data_dir = './data/cifar100'
    train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)
    train_sampler = DistributedSampler(train_dataset, num_replicas=num_replicas, rank=rank)
    train_dataloader = DataLoader(train_dataset, batch_size=128, sampler=train_sampler)
    val_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    return train_dataloader, val_loader

def load_cv_model(model_name, pretrained):
    model_path = f'./models/{model_name}.pth'
    if model_name=='resnet152':
        model = models.resnet152()
        model.load_state_dict(torch.load(model_path))
    if model_name=='resnet18':
        model = models.resnet18()
        model.load_state_dict(torch.load(model_path))
    if model_name=='resnet101':
        model = models.resnet101()
        model.load_state_dict(torch.load(model_path))
    if model_name=='vgg16':
        model = models.vgg16()
        torch.save(model.state_dict(), model_path)
    if model_name=='vgg19':
        model = models.vgg19(pretrained)
        torch.save(model.state_dict(), model_path)
    if model_name=='vit-base16':
        model = AutoModelForImageClassification.from_pretrained(model_path)  
    return model

