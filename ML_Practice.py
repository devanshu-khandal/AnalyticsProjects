
#%%
# import urllib.request
# from urllib.request import urlopen
# import ssl
# import json
# import os
# import certifi


import numpy as np
import pandas as pd
# from sklearn.datasets import load_diabetes
from sklearn.datasets import fetch_california_housing
# from sklearn.datasets import clear_data_home
# clear_data_home()  
import matplotlib.pyplot as plt

# In[1]:
# df = pd.read_csv("/Users/devanshu_khandal/Analytics_Projects/ML_DataSets/CaliforniaHousing/cal_housing.data"
#                 ,header=None)

df=fetch_california_housing()

df.columns = ["longitude","latitude","housingMedianAge","totalRooms","totalBedrooms","population","households","medianIncome","medianHouseValue"]

df.head()

# help(pd.read_csv)
# dataset = pd.DataFrame(data= np.c_[df['data'],df['target']],
#                      columns= df['feature_names'] + ['target'])
# dataset.head()

# In[2]:
df.describe()

# %%
