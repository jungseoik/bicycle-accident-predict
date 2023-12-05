import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics import ConfusionMatrix, Accuracy, Precision, Recall, F1Score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from copy import deepcopy
from tqdm import tqdm
from nn import *
import numpy as np
from train import *
#from metric import cm_to_metrics
from typing import List,Optional
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError 



def evaluate(
  model:nn.Module,
  criterion:callable,
  data_loader:DataLoader,
  device:str,
  metric:Optional[torchmetrics.metric.Metric]=None,
  multi_metrics: List[torchmetrics.metric.Metric]=None
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

  #if metric is not None:
    #metric.reset()
      
  with torch.inference_mode():
    for X, y in data_loader:
      X, y = X.to(device), y.to(device)
      output = model(X)
      total_loss += criterion(output, y).item() * len(y)
      if metric is not None:
        metric.update(output, y)
      if multi_metrics is not None:
        for metric in multi_metrics:
          metric.update(output, y)
  return total_loss/len(data_loader.dataset)


def kfold_cross_validate(model: nn.Module, 
                         criterion:callable, 
                         device:str, 
                         X_trn:np.array, y_trn:np.array, 
                         n_splits:int=5, 
                         lr = 0.01 , 
                         epochs = 100):
  print(X_trn.shape)
  print(y_trn.shape)
  
  if len(torch.unique(y_trn)) < n_splits:
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2023)
  else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=2023)
  #kf = KFold(n_splits=n_splits, shuffle=True, random_state=2023)
  #nets = [deepcopy(model) for i in range(n_splits)]
  nets = [ANN(input_dim=X_trn.shape[-1]).to(device) for i in range(n_splits)]

  scores = {
  'MSE': [],
  'MAE': [],
  'RMSE': [],
  "epochs_to_early_stop": []
  }

  # Convert tensor y_trn to NumPy array
  y_trn_np = y_trn.cpu().numpy()
  
  for i, (trn_idx, val_idx) in enumerate(kf.split(X_trn,y_trn_np)):
    X1, y1 = X_trn[trn_idx], y_trn[trn_idx]
  
    X_val, y_val = X_trn[val_idx], y_trn[val_idx]

    #DataLoader
    #ds = TensorDataset(X1, y1.squeeze().to(torch.float32))
    ds = TensorDataset(X1, y1.unsqueeze(1).to(torch.float32))
    #ds_val = TensorDataset(X_val, y_val.squeeze().to(torch.float32))
    ds_val = TensorDataset(X_val, y_val.unsqueeze(1).to(torch.float32))

    dl = DataLoader(ds, batch_size=256, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=len(ds_val), shuffle=False)

    net = nets[i].train()
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')

    patience=5
    pbar = tqdm(range(epochs)) 
    
    for j in pbar:
      optimizer = torch.optim.Adam(net.parameters(), lr=lr)
      mae, mse, rmse = MeanAbsoluteError().to(device), MeanSquaredError().to(device), MeanSquaredError(squared=False).to(device)
      loss = train_one_epoch(net, criterion, optimizer, dl, device)
      loss_val = evaluate(net, criterion, dl_val, device, multi_metrics = [mae, mse, rmse])
      mae_val, mse_val, rmse_val = mae.compute().item(), mse.compute().item(), rmse.compute().item()
      pbar.set_postfix(trn_loss=loss, val_loss=loss_val)
      
      if (j < epochs//2):
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
        epochs_to_early_stop = j - patience
        break

    # Metrics computation at the early stopping point

    scores["MAE"].append(mae_val)
    scores["MSE"].append(mse_val)
    scores["RMSE"].append(rmse_val)
    scores["epochs_to_early_stop"].append(epochs_to_early_stop)

    #opep.append(j-patience)
  # Convert history to a DataFrame
  #metrics_df = pd.DataFrame(history)
  
  # Calculate and append average metrics
  
  #history.append(average_metrics)

  mean_mae = sum(scores["MAE"]) / len(scores["MAE"])
  mean_mse = sum(scores["MSE"]) / len(scores["MSE"])
  mean_rmse = sum(scores["RMSE"]) / len(scores["RMSE"])
  mean_epochs_to_early_stop = sum(scores["epochs_to_early_stop"]) / len(scores["epochs_to_early_stop"])

  # Add mean values to the DataFrame
  scores["Mean_MAE"] = mean_mae
  scores["Mean_MSE"] = mean_mse
  scores["Mean_RMSE"] = mean_rmse
  scores["Mean_Epochs_to_Early_Stop"] = mean_epochs_to_early_stop

  metrics_df = pd.DataFrame(scores)
  # Save the metrics to a CSV file
  metrics_df.to_csv("metrics.csv", index=False)


'''
    bcm = ConfusionMatrix.to(device)
    evaluate(net, nn.functional.binary_cross_entropy, dl_val, device, bcm)
    history.append(bcm)
    opep.append(j-patience)

  return history


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

def get_args_parser(add_help=True):
  import argparse

  parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

  parser.add_argument("--data-train", default="./data/final.csv", type=str, help="train dataset path")
  # parser.add_argument("--data-test", default="./data/test.csv", type=str, help="test dataset path")
  parser.add_argument("--hidden-dim", default=128, type=int, help="dimension of hidden layer")
  parser.add_argument("--device", default="cpu", type=str, help="device (Use cpu/cuda/mps)")
  parser.add_argument("-b", "--batch-size", default=32, type=int, help="batch size")
  parser.add_argument("--do-shuffle", default=True, type=bool, help="shuffle")
  parser.add_argument("--epochs", default=300, type=int, metavar="N", help="number of total epochs to run")
  parser.add_argument("--lr", default=0.0001, type=float, help="learning rate")
  parser.add_argument("--pbar", default=True, type=bool, help="progress bar")
  parser.add_argument("-o", "--output", default="./model.pth", type=str, help="path to save output model")
  
  return parser

if __name__ == "__main__":
  args = get_args_parser().parse_args()
  main(args)

if __name__ == "__kfold_cross_validate__":
  args = get_args_parser().parse_args()
  kfold_cross_validate(args)

