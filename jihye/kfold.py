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
from metric import cm_to_metrics
from typing import Optional

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


def kfold_cross_validate(model: nn.Module, criterion:callable, device:str, X_trn:np.array, y_trn:np.array, n_splits:int=5, lr = 0.01 , epochs = 100):
  #print(X_trn)
  #print(y_trn)
  
  if len(torch.unique(y_trn)) < n_splits:
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2023)
  else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=2023)
  #kf = KFold(n_splits=n_splits, shuffle=True, random_state=2023)
  #nets = [deepcopy(model) for i in range(n_splits)]
  nets = [ANN(input_dim=X_trn.shape[-1]).to(device) for i in range(n_splits)]

  history = []

  # Define metrics
  confusion_matrix_metric = ConfusionMatrix(num_classes=3, task="multiclass")
  accuracy_metric = Accuracy(num_classes=3, task="multiclass")
  precision_metric = Precision(num_classes=3, average='weighted', task="multiclass")
  recall_metric = Recall(num_classes=3, average='weighted', task="multiclass")
  f1_metric = F1Score(num_classes=3, average='weighted', task="multiclass")

  # Convert tensor y_trn to NumPy array
  y_trn_np = y_trn.cpu().numpy()
  
  for i, (trn_idx, val_idx) in enumerate(kf.split(X_trn,y_trn_np)):
    X1, y1 = X_trn[trn_idx], y_trn[trn_idx]
  
    X_val, y_val = X_trn[val_idx], y_trn[val_idx]

    #DataLoader
    ds = TensorDataset(X1, y1.squeeze().to(torch.long))
    ds_val = TensorDataset(X_val, y_val.squeeze().to(torch.long))
    dl = DataLoader(ds, batch_size=256, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=len(ds_val), shuffle=False)

    #net = nets[i].train()
    net = nets[i].train()
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')

    patience=5
    pbar = tqdm(range(epochs)) 
    
    for j in pbar:
      optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
      accuracy = accuracy_metric
      loss = train_one_epoch(net, nn.CrossEntropyLoss(), optimizer, dl, device)
      loss_val = evaluate(net, nn.CrossEntropyLoss(), dl_val, device, accuracy)
      acc_val = accuracy.compute().item()
      pbar.set_postfix(trn_loss=loss, val_loss=loss_val, val_acc=acc_val)
      
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
    confusion_matrix_metric.to(device)
    evaluate(net, nn.functional.cross_entropy, dl_val, device, confusion_matrix_metric)
    accuracy_metric.to(device)
    evaluate(net, nn.functional.cross_entropy, dl_val, device, accuracy_metric)
    precision_metric.to(device)
    evaluate(net, nn.functional.cross_entropy, dl_val, device, precision_metric)
    recall_metric.to(device)
    evaluate(net, nn.functional.cross_entropy, dl_val, device, recall_metric)
    f1_metric.to(device)
    evaluate(net, nn.functional.cross_entropy, dl_val, device, f1_metric)

    history.append({
            "Fold": i + 1,
            "Accuracy": accuracy_metric.compute().item(),
            "Precision": precision_metric.compute().item(),
            "Recall": recall_metric.compute().item(),
            "F1": f1_metric.compute().item(),
            "Confusion_Matrix": confusion_matrix_metric.compute().numpy(),
            "Epochs_to_Early_Stop": epochs_to_early_stop
        })
  # Convert history to a DataFrame
  metrics_df = pd.DataFrame(history)
  
  # Calculate and append average metrics
  average_metrics = {
        "Fold": "Average",
        "Accuracy": metrics_df["Accuracy"].mean(),
        "Precision": metrics_df["Precision"].mean(),
        "Recall": metrics_df["Recall"].mean(),
        "F1": metrics_df["F1"].mean(),
        "Epochs_to_Early_Stop": metrics_df["Epochs_to_Early_Stop"].mean()
    }
  history.append(average_metrics)
  metrics_df = pd.DataFrame(history)
  # Save the metrics to a CSV file
  metrics_df.to_csv("metrics2.csv", index=False)


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

