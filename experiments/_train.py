
import os
import time
import pandas as pd
import cudf
import logging
import multiprocessing
from multiprocessing import Process, Value, Manager
from tqdm import tqdm
import warnings
import config as CONFIG

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score

import xgboost as xgb

warnings.filterwarnings("ignore")
multiprocessing.set_start_method("spawn", force=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class Model(object):
    def __init__(self):
        self.negetive_sample_ratio = 0.1
        self.negetive_sample_seed = 2023
        self.train_set = "/home/rapids/notebooks/workspace/dram/data/processed/train_set.csv"
        self.test_set = "/home/rapids/notebooks/workspace/dram/data/processed/test_set.csv"
        self.model = "/home/rapids/notebooks/workspace/dram/data/model/version1.model"
        self.features = ["server_manufacturer","DRAM_model","DIMM_number","e_row_2","e_row_4","e_row_8","e_row_16","e_row_32","e_col_2","e_col_4","e_col_8",\
                        "e_col_16","e_col_32","e_col_64","row_cnt","col_cnt","count","ce_storm_10",\
                        "ce_storm_20","ce_storm_24","ce_storm_30","ce_storm_40","ce_storm_50","e_bank_2_2",\
                        "e_bank_2_4","e_bank_2_8","e_bank_2_16","e_bank_2_32","e_bank_2_64","e_bank_4_2",\
                        "e_bank_4_4","e_bank_4_8","e_bank_4_16",\
                        "e_bank_4_32","e_bank_4_64","e_bank_8_2","e_bank_8_4","e_bank_8_8","e_bank_8_16",\
                        "e_bank_8_32","e_bank_8_64","e_bank_16_2","e_bank_16_4","\
                        e_bank_16_8","e_bank_16_16","e_bank_16_32","e_bank_16_64","e_bank_32_2","e_bank_32_4",\
                        "e_bank_32_8","e_bank_32_16","e_bank_32_32","e_bank_32_64"]

    def _get_features_one(self, idx, slice_features_one_list):
        logging.info(f"{idx} start")
        features_one = pd.read_csv(f"/home/rapids/notebooks/workspace/dram/data/processed/features_{idx}.csv")
        features_one = cudf.DataFrame(features_one)
        features_one["time_diff"] = cudf.to_datetime(features_one["failed_time"]) - cudf.to_datetime(features_one["error_time"])
        features_one["failed"] = features_one["time_diff"].dt.days <= 7 
        for u in features_one.columns:
            if features_one[u].dtype==bool:
                features_one[u]=features_one[u].astype('int')
        features_one_0 = features_one[features_one["failed"]==0].sample(frac=self.negetive_sample_ratio, random_state=self.negetive_sample_seed)
        features_one_1 = features_one[features_one["failed"]==1]
        features_dataset = cudf.concat([features_one_0, features_one_1], axis=0)
        features_dataset = features_dataset.sort_values(["sid", "error_time"])
        drop_list = ["error_time", "sid", "memoryid", "rankid", "bankid", "row", "col", "error_type", "failure_type", "failed_time", "time_diff", "failed"]
        features_with_failure_sampled_encoded_info = features_dataset[drop_list]
        for column in features_dataset.columns:
            if column in drop_list:
                features_dataset.drop(columns=[column], inplace=True)
        features_with_failure_sampled_encoded = cudf.get_dummies(features_dataset.iloc[:, 0:-1])
        features_with_failure_sampled_encoded = cudf.concat([features_with_failure_sampled_encoded_info, features_with_failure_sampled_encoded], axis=1)
        features_with_failure_sampled_encoded = features_with_failure_sampled_encoded.to_pandas()
        slice_features_one_list.append(features_with_failure_sampled_encoded)
        if "failed" not in features_with_failure_sampled_encoded.columns:
            logging.info("not failed")
        logging.info(f"{idx} finished")
        # self.train_dataset = pd.concat(features_with_failure_sampled_encoded_list, axis=0)

    def _get_dataset(self, stage="train", rerun=True):
        """
        stage: train | test
        rerun: True | False
        """

        if stage == "train":
            if not rerun and os.path.exists(self.train_set):
                self.train_set = pd.read_csv(self.train_set)
                logging.info("loading train set finished")
                return
            dataset_idx = [i for i in range(0, 23)]
        else:
            if not rerun and os.path.exists(self.test_set):
                self.test_set = pd.read_csv(self.test_set)
                logging.info("loading test set finished")
                return
            dataset_idx = [i for i in range(23, 30)]

        features_list = []
        alllens = 0
        while len(dataset_idx) > 0:
            manager = Manager()
            slice_features_one_list = manager.list()
            slice_idx = dataset_idx[:10]
            dataset_idx = dataset_idx[10:]
            logging.info(slice_idx)
            
            processes = []
            for idx in tqdm(slice_idx):
                if idx == 22:
                    continue
                p = Process(target=self._get_features_one, args=(idx, slice_features_one_list))
                processes.append(p)
                p.start()
            for p in processes:
                p.join()
            features_list.extend(slice_features_one_list)

        ret = pd.concat(features_list, axis=0)
        # ret = ret.fillna(-1)
        ret.to_csv(f"/home/rapids/notebooks/workspace/dram/data/processed/{stage}_set.csv")

        if stage == "train":
            self.train_set = ret
        else:
            self.test_set = ret

        logging.info(f"loading {stage} finished")

    def validation(self):
        self._get_dataset(stage="test", rerun=False)

        # self.train_y = self.train_set["failed"]
        # for column in self.train_set.columns:
        #     if "Unnamed" in column or "failed" in column:
        #         self.train_set = self.train_set.drop(columns=[column])
        # self.train_x = self.train_set

    def _training(self):
        self.train_set = self.train_set.fillna(-1)
        self.train_y = self.train_set["failed"]
        self.train_x = self.train_set.iloc[:, 14:]
        # for column in self.train_x.columns:
        #     if "Unnamed" in column or "failed" in column:
        #         self.train_set = self.train_set.drop(columns=[column])
        
        model = xgb.XGBClassifier()
        model.fit(self.train_x, self.train_y)
        model.save_model(self.model)

    def train(self):
        self._get_dataset(stage="train", rerun=False)
        self._training()
    

if __name__ == "__main__":
    t1 = time.time()
    model = Model()
    # model.train()
    model.validation()
    t2 = time.time()
    logging.info(f"{(t2 - t1) / 60} mins")