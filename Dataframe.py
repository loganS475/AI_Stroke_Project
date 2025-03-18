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
            nn.Linear(12, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(32, 8),
        )

    #shows how data flows through neural network
    def forward(self, x):
        x = self.flatten(x) #flattens data so it can enter neural network
        logits = self.linear_relu_stack(x)# runs data through layers and returns data unnormalized
        return logits
    
    
