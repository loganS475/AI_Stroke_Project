import pandas as pd
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms 



df = pd.read_csv('healthcare-dataset-stroke-data.csv')
df['bmi'] = df['bmi'].fillna(28.89)
training = df[:4089]
testing = df[4089:]
print(training)
print(testing)
