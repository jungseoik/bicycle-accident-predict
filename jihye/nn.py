import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.base import BaseEstimator
from tqdm.auto import tqdm
from torch.utils.data import TensorDataset

class ANN(nn.Module):
  def __init__(self, input_dim=107, hidden_dim=256, p=0.3):
    super().__init__()
    self.lin1 = nn.Linear(input_dim,hidden_dim)
    self.lin2 = nn.Linear(input_dim,hidden_dim)
    self.lin3 = nn.Linear(hidden_dim,3)
    self.dropout = nn.Dropout(p)
  def forward(self, x):
    x = self.lin1(x)
    x = nn.functional.relu(x)
    x = self.dropout(x)
    x = self.lin2(x)
    x = nn.functional.relu(x)
    x = self.dropout(x)
    x = self.lin3(x)
    x = nn.functional.softmax(x, dim=1)
    return x
  
class ANN_Estimator(ANN, BaseEstimator):
  def __init__(self, hidden=128, optim=torch.optim.Adam, lr=0.0001, loss_fn=nn.functional.binary_cross_entropy, device='cpu'):
    self.hidden = hidden
    self.optim = optim
    self.lr = lr
    self.loss_fn = loss_fn

    super().__init__(self.hidden)
    self.optimizer = self.optim(self.parameters(), lr=self.lr)
    self.device = device
    self.to(device)

  def fit(self, X, y):
    self.train()

    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=32, shuffle=True)

    pbar = tqdm(range(300))
    for _ in pbar:
      for _X, _y in dl:
        _X, _y = _X.to(self.device), _y.to(self.device)
        _pred = self.forward(_X)
        loss = self.loss_fn(_pred, _y.unsqueeze(-1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

      pbar.set_postfix(trn_loss=loss.item())

  def predict(self, X):
    self.eval()
    with torch.no_grad():
      pred = self.forward(torch.tensor(X, device=self.device))
    return (pred.cpu() > 0.5).float().numpy()