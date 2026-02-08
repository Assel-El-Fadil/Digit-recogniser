import pandas as pd
import numpy as np

data = pd.read_csv('train.csv')

data = np.array(data).T

labels = data[0]
pixels = data[1:]

