import os
import pandas as pd
from tqdm import tqdm

import config as CONFIG

class Process(object):
    def __init__(self):
        pass
    
    def run(self):
        self._read_data()
        self._store_data()
        
    def _read_data(self):
        df_list = []
        for i in [14, 15, 16]:
            df = pd.read_csv(os.path.join(CONFIG.PATH_PROCESSED, f"mcelog__{i}_large.csv"))
            df_list.append(df)
        self.mcelog = pd.concat(df_list, axis=0)
        
    def _store_data(self):
        slices = 16
        hosts = self.mcelog["sid"].drop_duplicates().to_list()
        slice_size = len(hosts) // slices
        host_slices = [hosts[i:i+slice_size] for i in range(0, len(hosts), slice_size)]

        for i, host_slice in tqdm(enumerate(host_slices)): 
            mcelog_one = self.mcelog[self.mcelog["sid"].isin(host_slice)]
            mcelog_one.to_csv(os.path.join(CONFIG.PATH_PROCESSED, f"mcelog_{i+14}.csv"))


if __name__ == "__main__":
    p = Process()
    p.run()