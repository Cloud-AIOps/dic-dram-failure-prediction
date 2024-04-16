import os
import time
import bisect
import logging
import pandas as pd
import multiprocessing
from datetime import datetime, timedelta
import multiprocessing
from multiprocessing import Process, Value, Queue, Pool, Manager
from functools import partial
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset, Dataset, SubsetRandomSampler
# import cupy
# import cudf

from tqdm import tqdm

multiprocessing.set_start_method("spawn", force=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def search_files(directory, includes=[], excludes=[], type="file") -> list:
    """
    :Param
    :type: dir | file
    """
    file_list = []
    for file in os.listdir(directory):
        if all(include in file for include in includes) and all(exclude not in file for exclude in excludes):
            absolute_file = os.path.join(directory, file)
            if type == "file" and os.path.isfile(absolute_file):
                file_list.append(absolute_file)
            if type == "dir" and os.path.isdir(absolute_file):
                file_list.append(absolute_file)
    return file_list
            
def add_new_features(df_cudf=None, stage="init"):
    """
    stage: init | extend
    """
    row_thresholds = [2, 4, 8, 16, 32]
    col_thresholds = [2, 4, 8, 16, 32, 64]
    if stage == "init":
        for th in row_thresholds:
            df_cudf[f"e_row_{th}"] = 0
        for th in col_thresholds:
            df_cudf[f"e_col_{th}"] = 0
        df_cudf["row_cnt"] = 0
        df_cudf["col_cnt"] = 0
    elif stage == "extend":
        for th in row_thresholds:
            df_cudf[f"e_row_{th}"] = df_cudf[f"row_cnt"] >= th
        for th in col_thresholds:
            df_cudf[f"e_col_{th}"] = df_cudf[f"col_cnt"] >= th
        for th_row in row_thresholds:
            for th_col in col_thresholds:
                df_cudf[f"e_bank_{th_row}_{th_col}"] = (df_cudf[f"e_row_{th_row}"] == 1) & (df_cudf[f"e_col_{th_col}"] == 1)
    else:
        logging.error("you have wrong 'stage' parameters")
    return df_cudf

def add_ce_storm_features(df_cudf=None):
    """
    """
    df_cudf["error_time"] = cudf.to_datetime(df_cudf["error_time"])
    df_cudf = df_cudf.set_index("error_time")
    df_cudf = df_cudf.fillna(-1)
    
    string_to_remove = "Unname"
    columns_to_keep = [col_name for col_name in df_cudf.columns if string_to_remove not in col_name]
    df_cudf = df_cudf[columns_to_keep]
    df_cudf["count"] = df_cudf["sid"].rolling("24h").apply("count")
    for ce_cnt in [10, 20, 24, 30, 40, 50]:
        df_cudf[f"ce_storm_{ce_cnt}"] = df_cudf["count"] >= ce_cnt
    df_cudf = df_cudf.reset_index()
    return df_cudf

def add_risky_features(df=None, column=None, new_column=None, window=24, threshold=2, time_column="timestamp"):
    """ """
    df_cudf = cudf.DataFrame(df)
    df_cudf = add_new_features(df_cudf, "init")
    
    host_cudf_list = []
    hosts = df_cudf["sid"].drop_duplicates().to_arrow().to_pylist()
    new_features = [column for column in df_cudf.columns if "e_row" in column or "e_col" in column]
    
    logging.info(len(df_cudf))
    
    for host in tqdm(hosts):
        host_cudf = df_cudf[df_cudf["sid"]==host]
        unique_set0 = set()
        unique_set1 = set()
        unique_cnt0 = [0] * len(host_cudf)
        unique_cnt1 = [0] * len(host_cudf)
        cursor = 0
                
        host_cudf["error_time"] = cudf.to_datetime(host_cudf["error_time"])
        for index, row in host_cudf.to_pandas().iterrows():
            row_column = int(row["row"])
            col_column = int(row["col"])
            unique_set0.add(row_column)
            unique_set1.add(col_column)
            unique_cnt0[cursor] = len(unique_set0)
            unique_cnt1[cursor] = len(unique_set1)
            cursor += 1

        host_cudf["row_cnt"] = unique_cnt0
        host_cudf["col_cnt"] = unique_cnt1
        
        host_cudf = add_ce_storm_features(host_cudf)
        
        host_cudf_list.append(host_cudf)
    
    df_cudf = cudf.concat(host_cudf_list, axis=0, ignore_index=True)
    df_cudf = add_new_features(df_cudf, "extend")
    
    # df_cudf[["sid","row","col","row_cnt","col_cnt", "e_row_2", "e_row_4", \
    #          "e_row_8", "e_col_2", "e_col_4", "e_col_8", \
    #          "e_bank_2_2", "e_bank_2_4", "e_bank_2_8", \
    #         "e_bank_4_2", "e_bank_4_4", "e_bank_4_8", \
    #         "e_bank_8_2", "e_bank_8_4", "e_bank_8_8", "ce_storm_10", "ce_storm_20", "ce_storm_30"]].to_csv("df_cudf.csv")
    # df_cudf[["sid","row","col", "count", "error_time", "ce_storm_10", "ce_storm_20", "ce_storm_30"]].to_csv("df_cudf.csv")
    return df_cudf.to_pandas()

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {"data": self.data[idx], "label": self.labels[idx]}
        return sample
        
if __name__ == "__main__":
    pass