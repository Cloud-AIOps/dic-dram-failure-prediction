import os
import json
import datetime
import logging
import warnings
import time

import pandas as pd
import numpy as np
from tqdm import tqdm
import cudf

import config as CONFIG
import utils

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Preprocess(object):
    def __init__(self):
        pass
        
    def run(self):
        self.read_data()
        self.update_data_time()
        self.split_data_into_small_slices()
    
    def read_data(self):
        logging.info("read data ...")
        self.mcelog = cudf.read_csv(os.path.join(CONFIG.PATH_RAW, "ali/mcelog.csv"))
        self.failure = cudf.read_csv(os.path.join(CONFIG.PATH_RAW, "ali/trouble_tickets.csv"))
        self.inventory = cudf.read_csv(os.path.join(CONFIG.PATH_RAW, "ali/inventory.csv"))
        logging.info("read data finished")
        
    def _format_date(self, date):
        y = int(date.split(" ")[0].split("-")[0])
        m = int(date.split(" ")[0].split("-")[1])
        d = date.split(" ")[0].split("-")[2]
        yy = "2020"
        if m <= 3:
            yy = "2019"
        m = (m + 9) % 12
        m = 12 if m == 0 else m
        mm = "0" + str(m) if m <= 9 else str(m)
        return f"{yy}-{mm}-{d} {date.split(' ')[1]}" 
    
    def _merge_one_with_extra_info(self, mcelog_one):
        mcelog_one = cudf.merge(mcelog_one, self.inventory, on=["sid"], how="left")
        failure_ue = self.failure[self.failure["failure_type"]==1]
        mcelog_one_tagged = cudf.merge(mcelog_one, failure_ue, on=["sid"], how="left")
        mcelog_one_tagged["failed_time"].fillna("2023-01-01 00:00:00", inplace=True)
        return mcelog_one_tagged
    
    def update_data_time(self):
        self.mcelog["error_time"] = self.mcelog["error_time"].apply(self._format_date)
        self.failure["failed_time"] = self.failure["failed_time"].apply(self._format_date)
        logging.info("updating data time finished")
        
    def split_data_into_small_slices(self):
        slices = 16
        hosts = self.mcelog["sid"].drop_duplicates().to_list()
        slice_size = len(hosts) // slices
        host_slices = [hosts[i:i+slice_size] for i in range(0, len(hosts), slice_size)]

        for i, host_slice in tqdm(enumerate(host_slices)): 
            mcelog_one = self.mcelog[self.mcelog["sid"].isin(host_slice)]
            mcelog_one_tagged = self._merge_one_with_extra_info(mcelog_one)
            mcelog_one_tagged.to_csv(os.path.join(CONFIG.PATH_PROCESSED, f"mcelog_{i}.csv"))

if __name__ == "__main__":
    pp = Preprocess()
    pp.run()