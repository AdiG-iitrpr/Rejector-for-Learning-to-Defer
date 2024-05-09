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
from torch.utils.data import Dataset, DataLoader
from util import AverageMeter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# class CustomDataset(Dataset):
#     def __init__(self, dataset, expert):
#         self.dataset = dataset
#         self.expert = expert

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):

#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         # Get input and target from the original dataset
#         input_data, target = self.dataset[idx]

#         # Assuming you have some function predict() that generates predictions for inputs
#         pred = generateHumanLabels(input_data, self.expert)  # Generate prediction for the current input

#         # Return input, target, and prediction as a sample
#         return {'input': input_data, 'target': target, 'pred': pred}


def generateHumanLabels(train_loader, expert):
    expert_pred = []
    for i, (input, target) in enumerate(train_loader):
        out = expert.predict(input, target)
        # out = torch.tensor(out)
        expert_pred.append(out)
    #     for j in range(len(target)):
    #         train_loader.dataset[j * train_loader.batch_size + j]['pred'] = out[j]
    return expert_pred
    # return out

def getData():
    dataset = 'cifar10'
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                        std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    transform_train = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
    train_dataset_all = datasets.__dict__[dataset.upper()]('./data', train=True, download=True,
                                                            transform=transform_train)
    train_size = int(0.90 * len(train_dataset_all))
    test_size = len(train_dataset_all) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(train_dataset_all, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=128, shuffle=False)
    return train_loader

def train_defer_model(epochs,model,model_defer,train_loader,expert_pred):
    # optimizer = torch.optim.SGD(model_defer.parameters(), lr=0.01)
    optimizer = torch.optim.Adam(model_defer.parameters(), lr=0.01)
    # criterion = BCELoss()
    criterion = nn.BCEWithLogitsLoss()
    loss = 0
    total = 0
    correct = 0
    model.eval()
    model_defer.train()
    for ep in range(epochs):
        batch_time = AverageMeter()
        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            correct = 0.0
            optimizer.zero_grad()
            target = target.to(device)
            input = input.to(device)
            output = model(input)
            defer_output = model_defer(output)
            final_defer_output = defer_output.squeeze()
            machine_final_class = torch.argmax(output, dim=1)
            defer_target = torch.zeros_like(machine_final_class)
            defer_target = defer_target.to(device)
            zeros = 0
            ones = 0
            for j in range(len(machine_final_class)):
                if machine_final_class[j] == target[j]:
                    defer_target[j] = 0
                    zeros += 1
                else:
                    if target[j] == expert_pred[i][j]:
                        defer_target[j] = 1
                        ones +=1 
                    else:
                        defer_target[j] = 0
                        zeros += 1
            # print(defer_target)
            # if(i%20==0): print(zero, one)
            loss = criterion(final_defer_output, defer_target.float())
            loss.backward()
            optimizer.step()
            final_defer_output = torch.round(torch.sigmoid(final_defer_output))
            correct = torch.sum(defer_target == final_defer_output).item()
            batch_time.update(time.time() - end)
            end = time.time()
            if(i%10==0):
                acc = ((float(correct)*100)/defer_target.shape[0])
                print('Epoch: [{0}] [{1}][{2}]\t'
                        'Loss {3}\t'
                        'Accuracy {4}\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Defer {5}'
                        .format(
                        ep, i, len(train_loader), round(loss.item(),4), round(acc,3), ones, batch_time=batch_time))
        # print(f"Epoch [{ep+1}/10], Loss: {loss.item():.4f}")

def main():
    # print(sys.argv[1])
    n_dataset = 10 #cifar10   
    model = WideResNet(28, n_dataset, 4, dropRate=0)
    baseline_model_type = sys.argv[1]
    model.load_state_dict(torch.load(baseline_model_type))
    model = model.to(device)
    model_defer = DeferModel()
    model_defer = model_defer.to(device)
    k = int(sys.argv[2])
    expert = synth_expert(k, 10)
    train_loader = getData()
    expert_pred = generateHumanLabels(train_loader, expert)
    # train_loader = generateHumanLabels(train_loader, expert)
    # custom_train_dataset = CustomDataset(train_loader, expert)
    # train_loader = DataLoader(custom_train_dataset, batch_size=128, shuffle=True)
    epochs = int(sys.argv[3])
    defer_model_location = sys.argv[4]
    # dump_file_location = defer_model_location+'.txt'
    # open(dump_file_location, 'w').close()
    # f = open(dump_file_location,"a")
    train_defer_model(epochs, model, model_defer, train_loader, expert_pred)
    # f.close()
    torch.save(model_defer.state_dict(), defer_model_location)

if __name__ == "__main__":
    main()

# run as python train_defer.py ./baseline_model_50(which defer model to use) 5(num of classes expert can predict) 10(epochs for training defer model) ./defer_model_50_5_10