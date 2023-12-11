import torch
import pandas as pd
from get import get,prepro
from factory_model import factory_model, create_models , models_cv
from cv import CV
from sklearn.metrics import accuracy_score
from nn import *


X_trn, y_trn, X_tst, y_tst = get('data/f5.csv')
models = create_models(X_trn, y_trn,'configW.json',4)
df = models_cv(models,X_trn,y_trn)
print(df.head())

df.to_csv("data/jihye.csv")

