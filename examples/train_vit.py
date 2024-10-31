import torch
import os
import torch.nn as nn
from engine import trainer, loader, pruner

import torch.optim as optim
import argparse
import time
import logging
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist


def parse():
    parser = argparse.ArgumentParser(description='PyTorch Vit-Base16 Training')
    parser.add_argument('--model_name', type=str, help='Option: vit-base')
    parser.add_argument('--dataset', type=str, help='Option: cifar10, cifar100, imagenet100')
    parser.add_argument('--log_file', type=str)
    parser.add_argument('--pruning_amount', type=float, default=0)
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--hook', type=str, default='default')
    parser.add_argument('--world_size', type=int)
    parser.add_argument('--rank', type=int)
    parser.add_argument('--dist_url', type=str)
    
    args = parser.parse_args()
    return args

args = parse()

logging.basicConfig(
    filename=args.log_file,
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

train_dataloader, val_dataloader, num_classes = loader.load_cv_data(args.dataset, args.world_size, args.rank)

dist.init_process_group(backend='nccl', init_method=args.dist_url, rank=args.rank, world_size=args.world_size)

device = torch.device("cuda")
# 加载模型
net = loader.load_cv_model(args.model_name, args.pretrained)
net.classifier = nn.Linear(net.classifier.in_features, num_classes)
net = DistributedDataParallel(net.to(device=device))

num_epochs = 10
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

prune = pruner.Pruner(args.pruning_amount)
train = trainer.VitTrainer(model=net, 
                        optimizer=optimizer,
                        criterion=criterion,
                        lr_scheduler=lr_scheduler,
                        device=device, train_dataloader=train_dataloader,
                        val_dataloader=val_dataloader, 
                        pruner=prune, 
                        comm_hook=args.hook)

start_time = time.time()
for epoch in range(num_epochs):
    avg_loss, accuracy = train.train_epoch(epoch)
    current_time = time.time() - start_time
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    logging.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}, Time: {current_time/60:4f}")