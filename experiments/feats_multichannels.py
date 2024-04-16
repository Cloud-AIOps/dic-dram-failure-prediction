import os
import re
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

banks = 15
channels = 3
models = len(CONFIG.DRAM_MODEL_MAPS)
max_row = 131072
max_col = 1024
row_block = 32
col_block = 16
row_section_size = max_row // row_block
col_section_size = max_col // col_block


# def _process_one(i):
#     lens = []
#     reals = []

#     df = pd.read_csv(os.path.join(CONFIG.PATH_PROCESSED, f"mcelog_{i}.csv"))
#     hosts = df["sid"].drop_duplicates().to_list()
#     for host in tqdm(hosts):
#         host_df = df[df["sid"] == host]
#         host_df = host_df.fillna(-1)
#         lens.append(len(host_df))

#         if host_df["failure_type"].sum() < 0:
#             reals.append(0)
#         else:
#             reals.append(1)
            
    
#     np.save(f"__lens_{i}.npy", lens)
#     np.save(f"__reals_{i}.npy", reals)

# def process_one(i):
#     df = pd.read_csv(os.path.join(CONFIG.PATH_PROCESSED, f"mcelog_{i}.csv"))
#     data = []
#     labels = []
#     hosts = df["sid"].drop_duplicates().to_list()

#     for host in tqdm(hosts):
#         host_matrix = np.zeros([models, channels, row_block, col_block])
#         host_df = df[df["sid"]==host]
#         host_df = host_df.fillna(-1)
        
#         label = (int(host_df["failure_type"].sum() < 0) + 1) % 2
#         labels.append(label)

#         host_df["row_section"] = host_df["row"] // row_section_size
#         host_df["col_section"] = host_df["col"] // col_section_size

#         # host_df_simple = host_df[["row_section", "col_section"]].drop_duplicates()

#         for _, row in host_df.iterrows():
#             host_matrix[int(CONFIG.DRAM_MODEL_MAPS[row["DRAM_model"]]), int(row["bankid"]) - 1, int(row["row_section"]), int(row["col_section"])] += 1
#         data.append([host_matrix])
    
#     np.save(os.path.join(CONFIG.PATH_PROCESSED, f"feats_{row_block}x{col_block}_with_times_{i}.npy"), data)
#     np.save(os.path.join(CONFIG.PATH_PROCESSED, f"labels_{row_block}x{col_block}_{i}.npy"), labels)    
    


# def do_feats():
#     dataset_idxs = [[i for i in range(0, 10)], [i for i in range(10, 20)], [i for i in range(20, 31)]]
#     for dataset_idx in dataset_idxs:
#         processes = []
#         for idx in tqdm(dataset_idx):
#             if idx == 22:
#                 continue
#             p = Process(target=process_one, args=(idx, ))
#             processes.append(p)
#             p.start()
#         for p in processes:
#             p.join()

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
    df = pd.read_csv(os.path.join(CONFIG.PATH_PROCESSED, f"mcelog_{i}.csv"))
    hosts = df["sid"].drop_duplicates().to_list()

    for host in tqdm(hosts):
        host_times = []
        host_data = []
        
        host_df = df[df["sid"]==host]
        host_df = host_df.fillna(-1)
        dram_model = host_df["DRAM_model"].iloc[0]
        host_matrix = np.full((channels, row_block, col_block), CONFIG.DRAM_MODEL_MAPS[dram_model])
        
        label = (int(host_df["failure_type"].sum() < 0) + 1) % 2
        # labels.append(label)

        host_df["row_section"] = host_df["row"] // row_section_size
        host_df["col_section"] = host_df["col"] // col_section_size

        # host_df_simple = host_df[["row_section", "col_section"]].drop_duplicates()

        host_d = {}
        for _, row in host_df.iterrows():
            host_matrix[(int(row["bankid"]) - 1) // (banks // channels), int(row["row_section"]), int(row["col_section"])] += 1 / CONFIG.DRAM_MODEL_SCALING_PARAM
            error_time = get_ceiling_time(row["error_time"])
            host_d[error_time] = [host_matrix]

        for k, v in host_d.items():
            host_times.append(k)
            host_data.append(v)
    
        np.save(os.path.join(CONFIG.PATH_PROCESSED_AGG, f"{host}_{label}_feats_{row_block}x{col_block}_multichannles.npy"), host_data)
        np.save(os.path.join(CONFIG.PATH_PROCESSED_AGG, f"{host}_{label}_times_{row_block}x{col_block}_multichannles.npy"), host_times)
        np.save(os.path.join(CONFIG.PATH_PROCESSED_AGG, f"{host}_{label}_labels_{row_block}x{col_block}_multichannles.npy"), label)

def do_feats_agg():
    dataset_idxs = [[i for i in range(0, 10)], [i for i in range(10, 20)], [i for i in range(20, 31)]]
    # dataset_idxs = [[i for i in range(0, 1)]]
    for dataset_idx in dataset_idxs:
        processes = []
        for idx in tqdm(dataset_idx):
            if idx == 22:
                continue
            p = Process(target=process_one_agg, args=(idx, ))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()

def merge():
    feats_files = utils.search_files(CONFIG.PATH_PROCESSED_AGG, includes=["Server", "multichannles", "feats"], excludes=[])
    host_data_list = []
    label_data_list = []
    for feats_file in tqdm(feats_files):
        pattern = r"Server_\d+"
        match = re.search(pattern, feats_file)
        host = match.group()
        pattern = r"_\d_"
        feats_file = feats_file.replace(host, "")
        match = re.search(pattern, feats_file)
        label = match.group()
        host_data = np.load(os.path.join(CONFIG.PATH_PROCESSED_AGG, f"{host}{label}feats_{row_block}x{col_block}_multichannles.npy"))
        label_data = np.load(os.path.join(CONFIG.PATH_PROCESSED_AGG, f"{host}{label}labels_{row_block}x{col_block}_multichannles.npy"))
        host_data_list.append(host_data[-1])
        label_data_list.append(label_data)
    np.save(os.path.join(CONFIG.PATH_PROCESSED, f"feats_{row_block}x{col_block}_with_times_multichannles.npy"), host_data_list)
    np.save(os.path.join(CONFIG.PATH_PROCESSED, f"labels_{row_block}x{col_block}_with_times__multichannles.npy"), label_data_list)

    
if __name__ == "__main__":
    t1 = time.time()
    # do_feats_agg()
    merge()
    t2 = time.time()
    logging.info(f"taking {(t2 - t1)/60} mins")