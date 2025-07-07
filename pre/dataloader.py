import numpy as np
import os
from glob import glob

class DataLoader:
    def __init__(self, data_path='./data'):
        self.data_path = data_path
        self.data = self.load_data()
        
    def load_data(self):
        data = {}
        for folder in glob(os.path.join(self.data_path, '*')):
            folder_name = os.path.basename(folder)
            data[folder_name] = {}
            for file in glob(os.path.join(folder, '*.csv')):
                channel = file.split('_')[-1].split('.')[0]
                temp = np.loadtxt(file, delimiter=',',skiprows=6,dtype=object)
                data[folder_name][int(channel)] = temp[:,8:].astype(float)
        return data
    
    def __call__(self):
        return self.data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, folder_name):
        return self.data[folder_name]
    
    def __iter__(self):
        return iter(self.data.keys())
    
    def __next__(self):
        return next(self.data.keys())
    