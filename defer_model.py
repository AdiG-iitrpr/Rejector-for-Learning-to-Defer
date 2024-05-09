import torch
import torch.nn as nn

class DeferModel(nn.Module):
    def __init__(self):
        super(DeferModel, self).__init__()
        # Number of input features is 10.
        self.layer_1 = nn.Linear(10, 64) 
        self.layer_2 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, 1) 

        # self.layer_1 = nn.Linear(10, 32) 
        # self.layer_2 = nn.Linear(32, 32)
        # self.layer_out = nn.Linear(32, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)
        # self.batchnorm1 = nn.BatchNorm1d(32)
        # self.batchnorm2 = nn.BatchNorm1d(32)
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        
        return x