import os
import json
import datetime
import logging
import warnings

import pandas as pd
import numpy as np

import config as CONFIG
import utils

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset, SubsetRandomSampler
from sklearn.metrics import precision_score, recall_score
import torch.nn as nn
import torch.nn.functional as F


class CNNModel_64x32(nn.Module):
    def __init__(self, conv1_kernel_size, conv2_kernel_size, conv3_kernel_size, conv4_kernel_size):
        super(CNNModel_64x32, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=conv1_kernel_size, padding=(conv1_kernel_size-1)//2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=conv2_kernel_size, padding=(conv2_kernel_size-1)//2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=conv3_kernel_size, padding=(conv3_kernel_size-1)//2)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=conv4_kernel_size, padding=(conv4_kernel_size-1)//2)

        self.fc1 = nn.Linear(256 * 4 * 2, 256) 
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(256, 2)  

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 256 * 4 * 2)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    

class CNNModel_32x16(nn.Module):
    def __init__(self, conv1_kernel_size, conv2_kernel_size, conv3_kernel_size, conv4_kernel_size):
        super(CNNModel_32x16, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=conv1_kernel_size, padding=(conv1_kernel_size-1)//2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=conv2_kernel_size, padding=(conv2_kernel_size-1)//2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=conv3_kernel_size, padding=(conv3_kernel_size-1)//2)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=conv4_kernel_size, padding=(conv4_kernel_size-1)//2)

        self.fc1 = nn.Linear(256 * 2 * 1, 256) 
        # self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(256, 2)  

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 256 * 2 * 1)
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.fc = nn.Linear(32 * 16 * 8, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    
# 定义基本的ResNet块
class BasicResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicResNetBlock, self).__init__()

        padding = (kernel_size - 1) // 2  

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(residual)  # 添加跳跃连接
        out = self.relu(out)
        return out

class ResNet4(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet4, self).__init__()

        self.in_channels = 16  

        self.conv1 = nn.Conv2d(1, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self.make_layer(BasicResNetBlock, 32, 2, stride=1, kernel_size=9)
        self.layer2 = self.make_layer(BasicResNetBlock, 64, 2, stride=1, kernel_size=7)
        self.layer3 = self.make_layer(BasicResNetBlock, 128, 2, stride=1, kernel_size=5)
        self.layer4 = self.make_layer(BasicResNetBlock, 256, 2, stride=1, kernel_size=3)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.MaxPool2d(1)

        self.fc = nn.Linear(256, num_classes)

    def make_layer(self, block, out_channels, num_blocks, stride, kernel_size):
        layers = []
        layers.append(block(self.in_channels, out_channels, kernel_size=kernel_size, stride=stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels, kernel_size=3, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x