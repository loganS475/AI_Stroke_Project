import pandas as pd
import numpy as np
import os
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('healthcare-dataset-stroke-data.csv')
#drops stroke column and enumerates categorical columns
df.loc[df["hypertension"]=="Yes","hypertension"] = 1
df.loc[df["hypertension"]=="No","hypertension"] = 0
df.loc[df["gender"]=="Male","gender"] = 1
df.loc[df["gender"]=="Female","gender"] = 0
df.loc[df['gender']=="Other","gender"]=-1
df.loc[df["ever_married"]=="Yes","ever_married"] = 1
df.loc[df["ever_married"]=="No","ever_married"] = 0
df.loc[df["work_type"]=="Govt_job", "work_type"] = 0
df.loc[df["work_type"]=="Private","work_type"] = 1
df.loc[df["work_type"]=="Self-employed", "work_type"] = -1
df.loc[df["work_type"]=="children", "work_type"] = 2
df.loc[df["work_type"]=="Never_worked","work_type"]= 3
df.loc[df["Residence_type"]=="Urban", "Residence_type"] = 0
df.loc[df["Residence_type"]=="Rural", "Residence_type"] = 1
df.loc[df["smoking_status"]=="never smoked", "smoking_status"] = 1
df.loc[df["smoking_status"]=="smokes", "smoking_status"] = -1
df.loc[df["smoking_status"]=="formerly smoked", "smoking_status"] = 0
df.loc[df["smoking_status"]=="Unknown", "smoking_status"] = 2
df.replace(np.nan,26.6,inplace=True)

#ensures that both training and testing will have people who have and haven't had a stroke
df.drop(columns=['id'],inplace=True)
x = df.drop(columns="stroke")
y= df['stroke']
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=57,test_size=0.25)

#normalizes the data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#create tensors for all dataframes
x_train_tensor = torch.from_numpy(x_train)
x_test_tensor = torch.from_numpy(x_test)
y_train_tensor = torch.from_numpy(y_train.to_numpy())
y_test_tensor = torch.from_numpy(y_test.to_numpy())

# Class for Neural Network
class NeuralNetwork():
    #sets up constructor for Neural Network
    def __init__(self,x):
        self.weights = torch.rand(x.shape[1],1,dtype = torch.float64,requires_grad=True) #trainable weights of model
        self.bias = torch.zeros(1,dtype=torch.float64,requires_grad = True) #intialized to zero

    #shows how data flows through neural network
    def forward(self, x):
        z = torch.matmul(x,self.weights) + self.bias
        y_pred = torch.sigmoid(z) # reduces output to (0,1]
        return y_pred
    #defines loss function for neural netowrk
    def loss_function(self,y_pred,y):
        loss = torch.mean((y_pred - y)**2)
        return loss

learning_rate = 0.01
epochs = 100

model = NeuralNetwork(x_test_tensor)

for epoch in range(epochs):
    y_pred = model.forward(x_train_tensor)

    loss = model.loss_function(y_pred,y_train_tensor)

    loss.backward()

    #Update Parameters
    with torch.no_grad():
        model.weights -= learning_rate * model.weights.grad
        model.bias -= learning_rate * model.bias.grad

    #Zero Gradients
    model.weights.grad.zero_()
    model.bias.grad.zero_()

#Analysis of results
with torch.no_grad():
  y_pred = model.forward(x_test_tensor)
  y_pred = (y_pred > 0.9).float()
  accuracy = (y_pred == y_test_tensor).float().mean()
  print(f'Accuracy: {accuracy.item()}')
     