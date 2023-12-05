import torch
from torch import nn
from torch.utils.data import DataLoader
import torchmetrics
from typing import Optional
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from nn import *
from preprocess import *
from kfold import *
from train import train_one_epoch
from imblearn.over_sampling import RandomOverSampler
from collections import Counter


data = pd.read_csv("data/pre.csv")

X = get_X(data)
y = get_y(data)[:,np.newaxis]

print(X.shape,y.shape)

#print(X.isnull().sum())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Current Device:", device)
#print(X.shape,y.shape)

X_trn, X_test, y_trn, y_test = split2(X,y)

#SMOTE 사용하기
sm = RandomOverSampler(random_state=42)
X_trn, y_trn = sm.fit_resample(X_trn, np.round(y_trn))

X_trn, X_test, y_trn, y_test = torch.tensor(X_trn), torch.tensor(X_test),torch.tensor(y_trn), torch.tensor(y_test)
X_trn, X_test, y_trn, y_test = X_trn.to(device), X_test.to(device), y_trn.to(device) , y_test.to(device)

print(X_trn.shape, y_trn.shape , X_trn.dtype, y_trn.dtype)

#Loader 올리기
ds = TensorDataset(X_trn, y_trn.squeeze().to(torch.long))
dl = DataLoader(ds, batch_size=128, shuffle=True)

model = ANN(X_trn.shape[-1],256).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

kfold_cross_validate(model,criterion=nn.MSELoss(), device=device, X_trn=X_trn, y_trn = y_trn, n_splits=5 , lr=0.001, epochs=100)