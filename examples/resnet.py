import torch
import os
import torch.nn as nn
from engine import trainer, loader, pruner

import torch.optim as optim
import argparse
import time
import logging
from torch.nn.parallel import DistributedDataParallel


def parse():
    parser = argparse.ArgumentParser(description='PyTorch ResNet Training')
    parser.add_argument('--model_name', type=str, help='Option: resnet18, resnet50, resnet152')
    parser.add_argument('--dataset', type=str, help='Option: cifar10, cifar100, imagenet100')
    parser.add_argument('--log_file', type=str)
    parser.add_argument('--pruning_amout', type=int, default=0)
    parser.add_argument('--pretrained', type=bool, default=False)
    parser.add_argument('--hook', type=str, default='default')
    args = parser.parse_args()
    return args

args = parse()

logging.basicConfig(
    filename=args.log_file,  # 输出日志到文件
    filemode='a',  # 追加模式
    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
    level=logging.INFO  # 日志级别
)

train_dataloader, val_dataloader, num_classes = loader.load_cv_data(args.dataset)

device = torch.device("cuda")
# 加载模型
net = loader.load_resnet(args.model_name, args.pretrained).to(device=device)
net.fc = nn.Linear(net.fc.in_features, num_classes)
net = DistributedDataParallel(net)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)


prune = pruner.Pruner(args.pruning_amout)
train = trainer.Trainer(model=net, 
                        optimizer=optimizer,
                        criterion=criterion,
                        lr_scheduler=lr_scheduler,
                        device=device, train_dataloader=train_dataloader,
                        val_dataloader=val_dataloader, 
                        pruner=prune, 
                        hook=args.hook)

num_epochs = 200
start_time = time.time()
for epoch in range(num_epochs):
    avg_loss, accuracy = train.train_epoch(epoch)
    current_time = time.time() - start_time
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    logging.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}, Time: {current_time:4f}")