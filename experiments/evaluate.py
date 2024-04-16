import os
import re
import json
import time
from datetime import datetime
import logging
import warnings

import pandas as pd
import numpy as np
from tqdm import tqdm

import wandb
import optuna

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from torch.utils.data import DataLoader, TensorDataset, Dataset, SubsetRandomSampler
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt

import model as md
import config as CONFIG
import utils

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MODEL_NAME = "resnet"

class Evaluate(object):
    def __init__(self):
        self.batch_size = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._read_data()
    
    def _read_data(self):
        test_data = []
        test_labels = []
        self.host_feats_d = {}
        self.host_times_d = {}
        self.host_label_d = {}
        file_list = utils.search_files(CONFIG.PATH_PROCESSED_AGG, includes=["_1_", "feats", ".npy"], excludes=[], type="file")
        time_list = utils.search_files(CONFIG.PATH_PROCESSED_AGG, includes=["_1_", "labels", ".npy"], excludes=[], type="file")
        for host_file in tqdm(file_list):
            pattern = r"Server_(\d+)"
            match = re.search(pattern, host_file)
            host = match.group(0)

            data = np.load(host_file).astype(np.float32)
            data = 2 / (1 + np.exp(-data)) - 1
            test_data.append(data[-1])
            test_labels.append(np.array(1.0))
            for i in range(len(data)):
                if host not in self.host_feats_d.keys():
                    self.host_feats_d[host] = []
                    self.host_label_d[host] = []
                else:
                    self.host_feats_d[host].append(data[i])
                    self.host_label_d[host].append(np.array(1.0))
        for host_file in tqdm(time_list):
            pattern = r"Server_(\d+)"
            match = re.search(pattern, host_file)
            host = match.group(0)

            data = np.load(host_file)
            for i in range(len(data)):
                if host not in self.host_times_d.keys():
                    self.host_times_d[host] = []
                else:
                    self.host_times_d[host].append(data[i])

        test_dataset = utils.CustomDataset(test_data, test_labels)

        logging.info(test_data[0].shape)
        logging.info(test_labels[0].shape)

        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        logging.info("loading data finished")

    def evaluate(self):
        model = md.ResNet4_32x16(conv1_kernel_size=9, conv2_kernel_size=5, \
                                  conv3_kernel_size=3, conv4_kernel_size=3)
        # model.to(self.device)
        # model = nn.DataParallel(model)
        
        logging.info(os.path.join(CONFIG.PATH_MODEL, f"{MODEL_NAME}-{CONFIG.MODEL_VERSION}"))
        model.load_state_dict(torch.load(os.path.join(CONFIG.PATH_MODEL, f"{MODEL_NAME}-{CONFIG.MODEL_VERSION}")))
        model.to(self.device)
        model.eval() 

        self.predictions = []
        self.true_labels = []
        """
        precision, recall on test set
        """
        correct = 0
        total = 0
        predictions = []
        true_labels = []
        with torch.no_grad():
            for batch in tqdm(self.test_loader):
                inputs, labels = batch["data"].to(self.device), batch["label"].to(self.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                predictions.extend(predicted.tolist())
                true_labels.extend(labels.tolist())
        
        logging.info(predictions)
        precision = precision_score(true_labels, predictions)
        recall = recall_score(true_labels, predictions)
        logging.info({"test precision": precision, "test recall": recall})

    def _get_gap_day_hour(self, time_str_0, time_str_1, time_str_2):
        first_time = datetime.strptime(time_str_0, "%Y-%m-%d %H:%M:%S")
        start_time = datetime.strptime(time_str_1, "%Y-%m-%d %H:%M:%S")
        end_time = datetime.strptime(time_str_2, "%Y-%m-%d %H:%M:%S")
        time_difference = end_time - start_time
        days = time_difference.days
        seconds_in_hour = 3600
        hours = (time_difference.seconds // seconds_in_hour) % 24

        first_time_difference = end_time - first_time
        first_days = first_time_difference.days
        first_seconds_in_hour = 3600
        first_hours = (first_time_difference.seconds // first_seconds_in_hour) % 24

        return f"{first_days} days {first_hours} hours - {days} days {hours} hours"


    def check_prediction_window(self):
        model = md.ResNet4_32x16(conv1_kernel_size=9, conv2_kernel_size=5, \
                                  conv3_kernel_size=3, conv4_kernel_size=3)
        logging.info(os.path.join(CONFIG.PATH_MODEL, f"{MODEL_NAME}-{CONFIG.MODEL_VERSION}"))
        model.load_state_dict(torch.load(os.path.join(CONFIG.PATH_MODEL, f"{MODEL_NAME}-{CONFIG.MODEL_VERSION}")))
        model.to(self.device)
        model.eval() 

        prediction_time_d = {}

        for host in tqdm(self.host_feats_d.keys()):
            feats = self.host_feats_d[host]
            label = self.host_label_d[host]
            test_dataset = utils.CustomDataset(feats, label)

            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

            correct = 0
            total = 0
            predictions = []
            true_labels = []
            with torch.no_grad():
                for batch in tqdm(test_loader):
                    inputs, labels = batch["data"].to(self.device), batch["label"].to(self.device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    predictions.extend(predicted.tolist())
                    true_labels.extend(labels.tolist())
            
            # logging.info(predictions)
            host_time_list = self.host_times_d[host]
            if 1 in predictions:
                index = predictions.index(1)
                # logging.info(f"{host}: {host_time_list[index]} - {host_time_list[-1]}")
                prediction_time_d[host] = self._get_gap_day_hour(host_time_list[0], host_time_list[index], host_time_list[-1], )
            else:
                pass
                # logging.info(f"{host}: not predicted")

        logging.info(len(prediction_time_d))
        with open("prediction_window.txt", "w+") as f:
            for k, v in prediction_time_d.items():
                logging.info(v)
                f.writelines(f"{k} \t {v}\n")
        # for k, v in prediction_time_d.items():
        #     logging.info(v)


if __name__ == "__main__":
    t1 = time.time()
    eva = Evaluate()
    # eva.evaluate()
    eva.check_prediction_window()
    t2 = time.time()
    logging.info(f"running time: {(t2 - t1) / 60} mins")
