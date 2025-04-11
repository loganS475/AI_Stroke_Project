import pandas as pd
import numpy as np
import os
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from torchvision import datasets, transforms 

df = pd.read_csv('healthcare-dataset-stroke-data.csv')
#drops stroke column and enumerates categorical columns
df.loc[df["hypertension"]=="Yes","hypertension"] = 1
df.loc[df["hypertension"]=="No","hypertension"] = 0
df.loc[df["gender"]=="Male","gender"] = 1
df.loc[df["gender"]=="Female","gender"] = 0
df.loc[df["ever_married"]=="Yes","ever_married"] = 1
df.loc[df["ever_married"]=="No","ever_married"] = 0
df.loc[df["work_type"]=="Govt_job", "work_type"] = 0
df.loc[df["work_type"]=="Private","work_type"] = 1
df.loc[df["work_type"]=="Self-employed", "work_type"] = -1
df.loc[df["work_type"]=="children", "work_type"] = 2
df.loc[df["Residence_type"]=="Urban", "Residence_type"] = 0
df.loc[df["Residence_type"]=="Rural", "Residence_type"] = 1
df.loc[df["smoking_status"]=="never smoked", "smoking_status"] = 1
df.loc[df["smoking_status"]=="smokes", "smoking_status"] = -1
df.loc[df["smoking_status"]=="formerly smoked", "smoking_status"] = 0
df.loc[df["smoking_status"]=="Unknown", "smoking_status"] = 2
df.replace(np.nan,26.6,inplace=True)

#Normalizing the data
mean_age = df['age'].mean()
std_age = df['age'].std()
mean_glu = df['avg_glucose_level'].mean()
std_glu = df['avg_glucose_level'].std()
mean_bmi = df['bmi'].mean()
std_bmi = df['bmi'].std()
for index,rows in df.iterrows():
    df.at[index,'age'] = (df.at[index,'age'] - mean_age)/std_age
    df.at[index,'avg_glucose_level'] = (df.at[index,'avg_glucose_level'] - mean_glu)/std_glu
    df.at[index,'bmi'] = (df.at[index,'bmi'] - mean_bmi)/std_bmi

#ensures that both training and testing will have people who have and haven't had a stroke
shuffled_df = df.sample(frac=1).reset_index(drop=True)
checkingdf = shuffled_df[["stroke"]].copy()
shuffled_df.drop(columns=['stroke'],inplace=True)
id_df = shuffled_df[['id']].copy()
shuffled_df.drop(columns=['id'],inplace=True)
training = shuffled_df[:4089]
testing = shuffled_df[4089:]
#with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #print(training)

#Checks if a Cuda device is avaliable otherwise defaults to CPU
if torch.accelerator.is_available():
    device = torch.accelerator.current_accelerator().type
else:
    device = "cpu"
print(f"Using {device} device")

# Class for Neural Network
class NeuralNetwork(nn.Module):
    #sets up constructor for Neural Network
    def __init__(self):
        super().__init__()# check to make sure constructor is properly set up
        self.flatten = nn.Flatten() # Reshapes data into a vector
        self.linear_relu_stack = nn.Sequential( #Builds hidden and visible layers of neural network
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    #shows how data flows through neural network
    def forward(self, x):
        x = self.flatten(x) #flattens data so it can enter neural network
        logits = self.linear_relu_stack(x)# runs data through layers and returns data unnormalized
        return logits #logits = unnormalized data

#Creates Neural Network Object   
model = NeuralNetwork().to(device)

#converts dataframe to torch tensor so model can be fed data.
training = training.apply(pd.to_numeric, errors ='coerce')
X_tensor = torch.tensor(training.values, dtype=torch.float32)
logits = model(X_tensor)
pred_probab = nn.Softmax(dim=1)(logits) #torch tensor output from neural network
temp = pred_probab.detach().numpy()
results = pd.DataFrame(temp)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(results)