import os
import json
import time
from datetime import datetime, timedelta
import logging
import warnings

import pandas as pd
import numpy as np
from tqdm import tqdm
# import cudf
from numba import cuda
from collections import Counter
# import cupy as cp

from sklearn.metrics import accuracy_score, precision_score, recall_score
import config as CONFIG
import utils

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.metrics import precision_score, recall_score
import torch.nn as nn
import torch.nn.functional as F

import multiprocessing
from multiprocessing import Process, Value, Manager

multiprocessing.set_start_method("spawn", force=True)


max_row = 131072
max_col = 1024
row_block = 32
col_block = 16
row_section_size = max_row // row_block
col_section_size = max_col // col_block

def get_ceiling_time(given_time_str):
    """
    2019-10-17 21:46:32 -> 2019-10-17 22:00:00
    """
    # given_time_str = "2019-10-17 23:46:32"
    given_time = datetime.strptime(given_time_str, "%Y-%m-%d %H:%M:%S")
    rounded_time = given_time + timedelta(hours=1) - timedelta(minutes=given_time.minute, seconds=given_time.second)
    rounded_time_str = rounded_time.strftime("%Y-%m-%d %H:%M:%S")
    return rounded_time_str


def process_one_agg(i):
    df = pd.read_csv(os.path.join(CONFIG.PATH_RAW, f"competition/seg/mcelog_{i}.csv"))
    hosts = df["serial_number"].drop_duplicates().to_list()

    for host in tqdm(hosts):
        host_times = []
        host_data = []
        host_matrix = np.zeros([row_block, col_block])
        host_df = df[df["serial_number"]==host]
        host_df = host_df.fillna(-1)
        
        label = (int(host_df["label"].sum() < 0) + 1) % 2
        # labels.append(label)

        host_df["row_section"] = host_df["row"] // row_section_size
        host_df["col_section"] = host_df["col"] // col_section_size

        # host_df_simple = host_df[["row_section", "col_section"]].drop_duplicates()

        host_d = {}
        for _, row in host_df.iterrows():
            host_matrix[int(row["row_section"]), int(row["col_section"])] += 1
            error_time = get_ceiling_time(row["collect_time"])
            host_d[error_time] = [host_matrix]

        for k, v in host_d.items():
            host_times.append(k)
            host_data.append(v)
    
        np.save(os.path.join(CONFIG.PATH_RAW, f"competition/agg/{host}_{label}_feats_{row_block}x{col_block}.npy"), host_data)
        np.save(os.path.join(CONFIG.PATH_RAW, f"competition/agg/{host}_{label}_labels_{row_block}x{col_block}.npy"), host_times)

def do_feats_agg():
    dataset_idxs = [[i for i in range(0, 10)], [i for i in range(10, 20)], [i for i in range(20, 31)]]
    # dataset_idxs = [[i for i in range(0, 1)]]
    for dataset_idx in dataset_idxs:
        processes = []
        for idx in tqdm(dataset_idx):
            p = Process(target=process_one_agg, args=(idx, ))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()


    
if __name__ == "__main__":
    t1 = time.time()
    """
    feats
    """
    do_feats_agg()


    t2 = time.time()
    logging.info(f"taking {(t2 - t1)/60} mins")