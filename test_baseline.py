import sys
import math
import torch
import torch.nn as nn
import random
import numpy as np
import torch.nn.functional as F
import argparse
import os
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
from defer_model import DeferModel
from expert import synth_expert
from util import Utility
from util import AverageMeter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def getData():
    dataset = 'cifar10'
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                        std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    transform_test = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
    test_dataset = datasets.__dict__[dataset.upper()]('./data', train=False, download=True,
                                                            transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=128, shuffle=False)
    return test_loader

def main():
    n_dataset = 10 #cifar10   
    model = WideResNet(28, n_dataset, 4, dropRate=0)
    baseline_model_type = sys.argv[1]
    model.load_state_dict(torch.load(baseline_model_type))
    model = model.to(device)
    test_loader = getData()
    accuracy = 0
    accuracy_sum = 0
    top1 = AverageMeter()
    utils = Utility()
    model.eval()
    for i, (input, target) in enumerate(test_loader):
        target = target.to(device)
        input = input.to(device)
        output = model(input)
        prec1 = utils.accuracy(output.data, target, topk=(1,))[0]
        top1.update(prec1.item(), input.size(0))
        print('Batch: [{0}]\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, top1=top1))
if __name__ == "__main__":
    main()

# run as python test_baseline.py ./baseline_model_10