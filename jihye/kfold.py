import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics import Accuracy, ConfusionMatrix
from sklearn.model_selection import KFold
from copy import deepcopy
from tqdm import tqdm
import numpy as np
from train import *
from metric import cm_to_metrics

def evaluate(
  model:nn.Module,
  criterion:callable,
  data_loader:DataLoader,
  device:str,
  metric:Optional[torchmetrics.metric.Metric]=None,
) -> float:
  '''evaluate
  
  Args:
      model: model
      criterions: list of criterion functions
      data_loader: data loader
      device: device
  '''
  model.eval()
  total_loss = 0.
  with torch.inference_mode():
    for X, y in data_loader:
      X, y = X.to(device), y.to(device)
      output = model(X)
      total_loss += criterion(output, y).item() * len(y)
      if metric is not None:
        metric.update(output, y)
  return total_loss/len(data_loader.dataset)


def kfold_cross_validate(model: nn.Module, criterion:callable, device:str, X_trn:np.array, y_trn:np.array, n_splits:int=5):
  
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
  kf = KFold(n_splits=n_splits, shuffle=True, random_state=2023)
  nets = [deepcopy(model) for i in range(n_splits)]
  history = []
  
  for i, (trn_idx, val_idx) in enumerate(kf.split(X_trn)):
    X, y = torch.tensor(X_trn[trn_idx]), torch.tensor(y_trn[trn_idx])
    X_val, y_val = torch.tensor(X_trn[val_idx]), torch.tensor(y_trn[val_idx])
    ds = TensorDataset(X, y)
    ds_val = TensorDataset(X_val, y_val)
    dl = DataLoader(ds, batch_size=32, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=len(ds_val), shuffle=False)

    net = nets[i].train()
    best_val_loss = float('inf')

    patience=5
    opep=[]
    pbar = tqdm(range(args.epochs)) 
    
    for j in pbar:
      accuracy = Accuracy.to(device)
      loss = train_one_epoch(net, nn.CrossEntropyLoss(), optimizer, dl, device)
      loss_val = evaluate(net, nn.CrossEntropyLoss(), dl_val, device, accuracy)
      acc_val = accuracy.compute().item()
      pbar.set_postfix(trn_loss=loss, val_loss=loss_val, val_acc=acc_val)
      
      if (j < args.epochs//2):
        continue
    # Save the best model based on validation loss
      if loss_val < best_val_loss:
        best_val_loss = loss_val
      
      # Reset early stopping counter
        early_stop_counter = 0
      else:
        # Increment early stopping counter
        early_stop_counter += 1

      # Check for early stopping
      if early_stop_counter >= patience:
        print(f'Early stopping at epoch {j}.')
        break

    bcm = ConfusionMatrix.to(device)
    evaluate(net, nn.functional.binary_cross_entropy, dl_val, device, bcm)
    history.append(bcm)
    opep.append(j-patience)

  return history

'''
# k-fold cross validation
  scores, opep= kfold_cross_validate(model, loss_func, device, X_trn, y_trn)
  mean_scores = {k:sum(v) / len(v) for k, v in scores.items()}
  optimalepoch = sum(opep)//len(opep)
  print(f"{mean_scores} / optimal epoch = {optimalepoch}")

  # train with full trainset
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
  pbar = range(optimalepoch)
  if args.pbar:
    pbar = tqdm(pbar)
  for _ in pbar:
    loss = train_one_epoch(model, loss_func, optimizer, dl, device)
    pbar.set_postfix(trn_loss=loss)
'''