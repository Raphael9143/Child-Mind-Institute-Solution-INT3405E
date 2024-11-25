import pandas as pd 
import os

data_path = 'data/'

train = 'train.csv'

print(pd.read_csv(os.path.join(data_path, train)).head())