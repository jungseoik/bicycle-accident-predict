import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def fuck(train,test, target_column, batch_size=64, test_size=0.2, random_state=42):
 
    # 입력 변수(X)와 출력 변수(y) 나누기
    X_trn = train.drop(target_column, axis=1)
    y_trn = train[target_column]
    X_tst = test
    
    # PyTorch Tensor로 변환
    X_train_tensor = torch.tensor(X_trn, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_trn.values, dtype=torch.float32).view(-1, 1)  # 모델의 출력이 1차원이어야 하므로 reshape
    X_test_tensor = torch.tensor(X_tst, dtype=torch.float32)
    
    # DataLoader 생성
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return X_train_tensor,  X_test_tensor, y_train_tensor , train_dataloader


