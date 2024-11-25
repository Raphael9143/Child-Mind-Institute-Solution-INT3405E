import pandas as pd 
import os
import numpy as np


data_path = 'data/'

train = 'train.csv'

print(pd.read_csv(os.path.join(data_path, train)).head())