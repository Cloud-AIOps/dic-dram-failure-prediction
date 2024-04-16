import os
import json
import datetime
import logging
import warnings

import pandas as pd
import numpy as np
from tqdm import tqdm
from numba import cuda
from collections import Counter

from sklearn.metrics import accuracy_score, precision_score, recall_score
import config as CONFIG
# import utils

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

from torchsummary import summary
from torchviz import make_dot

from tensorboardX import SummaryWriter

def train():
    
    dataset_list = []


    for i in range(31):
        if i == 22:
            continue
        data = np.load(f"_feats_{i}.npy")
        label = np.load(f"_labels_{i}.npy")

        data = torch.tensor(data, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.int64)

        dataset = TensorDataset(data, label)
        dataset_list.append(dataset)

        datasets = torch.utils.data.ConcatDataset(dataset_list)
    
    writer = SummaryWriter()
    
    torch.manual_seed(42)
    np.random.seed(42)

    import torch.nn as nn
    import torch.nn.functional as F

    class CNNModel(nn.Module):
        def __init__(self):
            super(CNNModel, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(32 * 16 * 8, 64)  
            self.fc2 = nn.Linear(64, 2)  

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = x.view(-1, 32 * 16 * 8)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # # 创建自定义数据集
    # class CustomDataset(Dataset):
    #     def __init__(self, X, y):
    #         self.X = X
    #         self.y = y

    #     def __len__(self):
    #         return len(self.X)

    #     def __getitem__(self, idx):
    #         return self.X[idx], self.y[idx]

    # # 创建数据加载器
    # dataset = CustomDataset(X, y)
    batch_size = 32
    data_loader = DataLoader(datasets, batch_size=batch_size, shuffle=True)


    # 定义训练集和测试集的样本比例
    train_ratio = 0.9  # 训练集占总数据的80%
    test_ratio = 1 - train_ratio  # 测试集占总数据的20%

    # 计算样本数量
    total_samples = len(data_loader.dataset)

    # 计算训练集和测试集的样本数量
    train_size = int(train_ratio * total_samples)
    test_size = total_samples - train_size

    # 创建随机划分的索引
    indices = list(range(total_samples))
    import random
    random.seed(42)
    random.shuffle(indices)

    # 根据划分的样本数量，划分训练集和测试集的索引
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    # 创建训练集和测试集的采样器
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # 创建训练集和测试集的 DataLoader
    train_loader = DataLoader(data_loader.dataset, batch_size=64, sampler=train_sampler)
    test_loader = DataLoader(data_loader.dataset, batch_size=64, sampler=test_sampler)



    # 创建模型实例
    model = CNNModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("adfasdf", device)
    model.to(device)

    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数适用于分类问题
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器

    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        for idx, (_inputs, _labels) in enumerate(train_loader):
            _inputs, _labels = _inputs.to(device), _labels.to(device)
            optimizer.zero_grad()
            outputs = model(_inputs.unsqueeze(1))  # 在通道维度上增加1
            
            # dot = make_dot(outputs, params=dict(model.named_parameters()))
            # dot.render("___cnn_model_arch", format="png")  # 将模型图保存为PNG文件

            
            loss = criterion(outputs, _labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            # writer.add_scalar('Train Loss', loss.item(), epoch * len(train_loader)+idx)

            # 记录每个epoch的平均训练损失
        average_loss = running_loss / len(train_loader)
        # writer.add_scalar('Average Train Loss', average_loss, epoch)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(data_loader)}')

    writer.close()
    print('Finished Training')


    model.eval()
    correct = 0
    total = 0

    predictions = []
    true_labels = []

    with torch.no_grad():
        for _inputs, _labels in train_loader:
            _inputs, _labels = _inputs.to(device), _labels.to(device)
            outputs = model(_inputs.unsqueeze(1))
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.tolist())
            true_labels.extend(_labels.tolist())

    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)

    print(f'Precision: {precision}')
    print(f'Recall: {recall}')


if __name__ == "__main__":
    train()