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
    def __init__(self, conv1_kernel_size, conv2_kernel_size, conv3_kernel_size, conv4_kernel_size, dropout, conv1_out_channels, conv2_out_channels,\
                                  conv3_out_channels, conv4_out_channels):
        super(CNNModel_64x32, self).__init__()
        
        self.conv4_out_channels = conv4_out_channels
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=conv1_out_channels, kernel_size=conv1_kernel_size, padding=(conv1_kernel_size-1)//2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=conv1_out_channels, out_channels=conv2_out_channels, kernel_size=conv2_kernel_size, padding=(conv2_kernel_size-1)//2)
        self.conv3 = nn.Conv2d(in_channels=conv2_out_channels, out_channels=conv3_out_channels, kernel_size=conv3_kernel_size, padding=(conv3_kernel_size-1)//2)
        self.conv4 = nn.Conv2d(in_channels=conv3_out_channels, out_channels=conv4_out_channels, kernel_size=conv4_kernel_size, padding=(conv4_kernel_size-1)//2)

        self.fc1 = nn.Linear(self.conv4_out_channels * 4 * 2, 512) 
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, 2)  

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, self.conv4_out_channels * 4 * 2)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    

class CNNModel_32x16(nn.Module):
    def __init__(self, conv1_kernel_size, conv2_kernel_size, conv3_kernel_size, conv4_kernel_size, dropout, conv1_out_channels, conv2_out_channels,\
                                  conv3_out_channels, conv4_out_channels, channel=1):
        super(CNNModel_32x16, self).__init__()
        
        self.conv4_out_channels = conv4_out_channels
        
        self.conv1 = nn.Conv2d(in_channels=channel, out_channels=conv1_out_channels, kernel_size=conv1_kernel_size, padding=(conv1_kernel_size-1)//2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=conv1_out_channels, out_channels=conv2_out_channels, kernel_size=conv2_kernel_size, padding=(conv2_kernel_size-1)//2)
        self.conv3 = nn.Conv2d(in_channels=conv2_out_channels, out_channels=conv3_out_channels, kernel_size=conv3_kernel_size, padding=(conv3_kernel_size-1)//2)
        self.conv4 = nn.Conv2d(in_channels=conv3_out_channels, out_channels=conv4_out_channels, kernel_size=conv4_kernel_size, padding=(conv4_kernel_size-1)//2)

        self.fc1 = nn.Linear(conv4_out_channels * 2 * 1, 256) 
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, 2)  

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, self.conv4_out_channels * 2 * 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CNNModel_32x16_two(nn.Module):
    def __init__(self, channel, conv1_kernel_size, conv2_kernel_size, dropout, conv1_out_channels, conv2_out_channels):
        super(CNNModel_32x16_two, self).__init__()
        
        self.conv2_out_channels = conv2_out_channels
        
        self.conv1 = nn.Conv2d(in_channels=channel, out_channels=conv1_out_channels, kernel_size=conv1_kernel_size, padding=(conv1_kernel_size-1)//2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=conv1_out_channels, out_channels=conv2_out_channels, kernel_size=conv2_kernel_size, padding=(conv2_kernel_size-1)//2)

        self.fc1 = nn.Linear(conv2_out_channels * 8 * 4, 64) 
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 2)  

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.conv2_out_channels * 8 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
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
    
"""
ResNet
"""
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

        out += self.shortcut(residual)
        out = self.relu(out)
        return out

class ResNet4_32x16(nn.Module):
    def __init__(self, conv1_kernel_size, conv2_kernel_size, conv3_kernel_size, conv4_kernel_size, dropout=0.2, channel=1):
        super(ResNet4_32x16, self).__init__()

        self.in_channels = 32

        self.conv1 = nn.Conv2d(channel, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self.make_layer(BasicResNetBlock, 32, 2, stride=1, kernel_size=conv1_kernel_size)
        self.layer2 = self.make_layer(BasicResNetBlock, 64, 2, stride=1, kernel_size=conv2_kernel_size)
        self.layer3 = self.make_layer(BasicResNetBlock, 128, 2, stride=1, kernel_size=conv3_kernel_size)
        self.layer4 = self.make_layer(BasicResNetBlock, 256, 2, stride=1, kernel_size=conv4_kernel_size)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(256, 2)

    def make_layer(self, block, out_channels, num_blocks, stride, kernel_size):
        layers = []
        layers.append(block(self.in_channels, out_channels, kernel_size=kernel_size, stride=stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels, kernel_size=kernel_size, stride=1))
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
        # x = self.dropout(x)
        x = self.fc(x)
        return x

class ResNet4_32x16_Two(nn.Module):
    def __init__(self, conv1_kernel_size, conv2_kernel_size, conv3_kernel_size, conv4_kernel_size, dropout=0.2, channel=1):
        super(ResNet4_32x16_Two, self).__init__()

        self.in_channels = 32

        self.conv1 = nn.Conv2d(channel, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self.make_layer(BasicResNetBlock, 32, 2, stride=1, kernel_size=conv1_kernel_size)
        self.layer2 = self.make_layer(BasicResNetBlock, 64, 2, stride=1, kernel_size=conv2_kernel_size)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(64, 2)
        

    def make_layer(self, block, out_channels, num_blocks, stride, kernel_size):
        layers = []
        layers.append(block(self.in_channels, out_channels, kernel_size=kernel_size, stride=stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels, kernel_size=kernel_size, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class ResNet4_32x16_multichannels(nn.Module):
    def __init__(self, conv1_kernel_size, conv2_kernel_size, conv3_kernel_size, conv4_kernel_size):
        super(ResNet4_32x16_multichannels, self).__init__()

        self.in_channels = 32

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self.make_layer(BasicResNetBlock, 32, 2, stride=1, kernel_size=conv1_kernel_size)
        self.layer2 = self.make_layer(BasicResNetBlock, 64, 2, stride=1, kernel_size=conv2_kernel_size)
        self.layer3 = self.make_layer(BasicResNetBlock, 128, 2, stride=1, kernel_size=conv3_kernel_size)
        self.layer4 = self.make_layer(BasicResNetBlock, 256, 2, stride=1, kernel_size=conv4_kernel_size)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 2)

    def make_layer(self, block, out_channels, num_blocks, stride, kernel_size):
        layers = []
        layers.append(block(self.in_channels, out_channels, kernel_size=kernel_size, stride=stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels, kernel_size=kernel_size, stride=1))
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


"""
ViT
"""
class ViT(nn.Module):
    def __init__(self, image_size, num_classes, patch_size, hidden_dim, num_heads, num_layers):
        super(ViT, self).__init__()
        self.embedding = nn.Conv2d(image_size[0], hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0), x.size(1), -1)  # Flatten spatial dimensions
        x = x.permute(2, 0, 1)  # Adjust the dimension order
        x = self.transformer(x, x)  # Self-attention with source and target being the same
        x = x.mean(dim=0)  # Global average pooling
        x = self.fc(x)
        return x