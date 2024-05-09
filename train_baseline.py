import math
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import random
import shutil
import time
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from baseline_model import WideResNet
from util import AverageMeter
from util import Utility

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def my_CrossEntropyLoss(outputs, labels):
    batch_size = outputs.size()[0]  # batch_size
    outputs = - torch.log2(outputs[range(batch_size), labels])  # pick the values corresponding to the labels
    return torch.sum(outputs) / batch_size

def train_classifier(train_loader, model, optimizer, scheduler, epoch, expert_fn, n_classes, utils):
    """Train for one epoch on the training set"""
    # expertfn: a number here k 
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        target = target.to(device)
        input = input.to(device)

        # compute output
        output = model(input)
        # compute loss
        loss = my_CrossEntropyLoss(output, target)

        # measure accuracy and record loss
        prec1 = utils.accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses, top1=top1))

def run_classifier(model, data_aug, n_dataset, expert_fn, epochs, utils):
    global best_prec1
    # Data loading code
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    if data_aug:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                              (4, 4, 4, 4), mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    if n_dataset == 10:
        dataset = 'cifar10'
    elif n_dataset == 100:
        dataset = 'cifar100'

    kwargs = {'num_workers': 0, 'pin_memory': True}

    train_dataset_all = datasets.__dict__[dataset.upper()]('./data', train=True, download=True,
                                                           transform=transform_train)
    train_size = int(0.90 * len(train_dataset_all))
    test_size = len(train_dataset_all) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(train_dataset_all, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=128, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=128, shuffle=True, **kwargs)

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    # model = torch.nn.DataParallel(model).cuda()
    model = model.to(device)

    # optionally resume from a checkpoint

    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    optimizer = torch.optim.SGD(model.parameters(), 0.1,
                                momentum=0.9, nesterov=True,
                                weight_decay=5e-4)

    # cosine learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * 200)

    for epoch in range(0, epochs):
        # train for one epoch
        train_classifier(train_loader, model, optimizer, scheduler, epoch, expert_fn, n_dataset, utils)
        if(epoch==0): torch.save(model.state_dict(), './baseline_model_1')
        if(epoch==9): torch.save(model.state_dict(), './baseline_model_10')
        if(epoch==49): torch.save(model.state_dict(), './baseline_model_50')
        if(epoch==99): torch.save(model.state_dict(), './baseline_model_100')
        if(epoch==149): torch.save(model.state_dict(), './baseline_model_150')
        if(epoch==199): torch.save(model.state_dict(), './baseline_model_200')


def main():
    utils = Utility()
    n_dataset = 10  # cifar-10, 100 for cifar-100
    model_classifier = WideResNet(28, n_dataset, 4, dropRate=0)
    num_epochs = int(sys.argv[1])
    # baseline_model_location = sys.argv[2]
    run_classifier(model_classifier, False, n_dataset, 0, num_epochs, utils)
    # torch.save(model_classifier.state_dict(), baseline_model_location)

if __name__ == "__main__":
    main()


# Run it as python train_baseline.py epochs location_data