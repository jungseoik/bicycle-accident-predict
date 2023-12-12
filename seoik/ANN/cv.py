from tqdm.auto import tqdm
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import mean_squared_error
import numpy as np
from copy import deepcopy
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import torch
import math
import matplotlib.pyplot as plt
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError
from typing import Optional, List
import torchmetrics
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch.nn as nn
from torch.utils.data import DataLoader

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
  model.model.eval()
  total_loss = 0.
  true_labels = []
  pred_labels = []
  with torch.inference_mode():
    for X, y in data_loader:
      X, y = X.to(device), y.to(device)
      output = model.model(X)
      total_loss += criterion(output, y).item() * len(y)
      if metric is not None:
        metric.update(output, y)

      true_labels.extend(y.cpu().numpy())
      pred_labels.extend(torch.argmax(output, dim=1).cpu().numpy())
  
  accuracy = accuracy_score(true_labels, pred_labels)
  precision_per_class = precision_score(true_labels, pred_labels, average=None)
  recall_per_class = recall_score(true_labels, pred_labels, average=None)
  f1_per_class = f1_score(true_labels, pred_labels, average=None)

  confusion_matrix1 = multilabel_confusion_matrix(true_labels, pred_labels)
  confusion_matrix2 = confusion_matrix(true_labels, pred_labels)


  '''
  accuracy = accuracy_score(true_labels, pred_labels)
  precision_per_class = precision_score(true_labels, pred_labels, average='micro')
  recall_per_class = recall_score(true_labels, pred_labels, average='micro')
  f1_per_class = f1_score(true_labels, pred_labels, average='micro')
  confusion_matrix1 = confusion_matrix(true_labels, pred_labels)
  
  metrics_df = pd.DataFrame({
        'Accuracy': [accuracy],
        'Precision (Class 0)': [precision_per_class],
        'Recall (Class 0)': [recall_per_class],
        'F1 Score (Class 0)': [f1_per_class],
        'Confusion Matrix': [confusion_matrix1]  # Convert to list for DataFrame
    })
  
'''


  
  # Create a DataFrame
  metrics_df = pd.DataFrame({
    'Accuracy': [accuracy],
    'Precision (Class 0)': [precision_per_class[0]],
    'Precision (Class 1)': [precision_per_class[1]],
    'Precision (Class 2)': [precision_per_class[2]],
    'Recall (Class 0)': [recall_per_class[0]],
    'Recall (Class 1)': [recall_per_class[1]],
    'Recall (Class 2)': [recall_per_class[2]],
    'F1 Score (Class 0)': [f1_per_class[0]],
    'F1 Score (Class 1)': [f1_per_class[1]],
    'F1 Score (Class 2)': [f1_per_class[2]],
    'Label Confusion Matrix': [confusion_matrix1],
    'Confusion Matrix': [confusion_matrix2]
  })
  

  # Save to CSV
  #metrics_df.to_csv('evaluation_metrics.csv', index=False)

  # Print the results
  #print(metrics_df)
  #print("Evaluation metrics saved to 'evaluation_metrics.csv'")

  return total_loss/len(data_loader.dataset), metrics_df



def ClfiCV(model, feature, label, n_splits=5):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    model_performance = []


    patience=5 #얼리스탑핑 용

    print(f"\n=== {model.__class__.__name__} ===")
    performance_metrics = {
        'Model': model.__class__.__name__, 
        
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],

        ##추가한 부분
        'confusion_matrix': []
        # 'early_stopping_epoch': []
        ##추가한 부분
    }

    nets = [deepcopy(model) for _ in range(n_splits)]

    ##얼리스탑핑 위해 추가한부분
    # best_loss = np.inf
    # no_improve_count = 0
    ##얼리스탑핑 위해 추가한부분

    for i, (train_index, test_index) in enumerate(kfold.split(feature, label)):
        x_train, x_test = feature[train_index], feature[test_index]
        y_train, y_test = label[train_index], label[test_index]

        net = nets[i]
        net.fit(x_train, y_train)

        pred = net.predict(x_test)

        ##혼동행렬 위해 추가
        cm = confusion_matrix(y_test, pred)
        print(f'\n#{i+1} Confusion Matrix: \n{cm}')
        performance_metrics['confusion_matrix'].append(cm)
        ##혼동행렬 위해 추가

        try:
            accuracy = accuracy_score(y_test, pred)
            precision = precision_score(y_test, pred, average='macro')
            recall = recall_score(y_test, pred, average='macro')
            f1 = f1_score(y_test, pred, average='macro')
        except Exception as e:
            print("Error occurred:", str(e))
            accuracy, precision, recall, f1 = np.nan, np.nan, np.nan, np.nan

        print(f'\n#{i+1} 교차 검증 Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}')
        # print('#{0} 검증 세트 인덱스 : {1}'.format(i+1, test_index))

        performance_metrics['accuracy'].append(accuracy)
        performance_metrics['precision'].append(precision)
        performance_metrics['recall'].append(recall)
        performance_metrics['f1'].append(f1)
        
        # Early stopping 추가한부분
        # loss = 1 - accuracy  # assuming you want to minimize error = 1 - accuracy
        # if loss < best_loss:
        #     best_loss = loss
        #     no_improve_count = 0
        # else:
        #     no_improve_count += 1
        #     if no_improve_count >= patience:
        #         print("Early stopping at epoch:", i+1)
        #         performance_metrics['early_stopping_epoch'].append(i+1)
        #         break
        # Early stopping 추가한부분
    i=0

    avg_performance = {
        'Model': performance_metrics['Model'] + str(i),
        'Avg Accuracy': np.mean(performance_metrics['accuracy']),
        'Avg Precision': np.mean(performance_metrics['precision']),
        'Avg Recall': np.mean(performance_metrics['recall']),
        'Avg F1': np.mean(performance_metrics['f1']),
    
        ##추가한 부분
        'Avg Confusion Matrix': np.mean(performance_metrics['confusion_matrix'], axis=0).astype(int)
        
        # 'Avg Early Stopping Epoch': np.mean(performance_metrics['early_stopping_epoch'])

        #오버피팅이 일어나서 오버피팅이 일어난걸 개선하겠다라는 목적성이 있으면 적합하다
        #오버피팅이 일어난다는건 파이프라인이 제대로 돌아가고 학습이 잘 돌아가고 있다 라는걸 확인한 후
        #나머지 하이퍼 파라미터 다 사용해봤는데 오버피팅을 못잡으면 얼리스탑핑을 사용한다.

        #오버피팅 -> 모델파라미터가 너무 많을때 발생한다.
    }
    model_performance.append(avg_performance)
    i+=1
    
    print('\n## 평균 검증 Accuracy:', np.mean(performance_metrics['accuracy']))
    print('## 평균 검증 Precision:', np.mean(performance_metrics['precision']))
    print('## 평균 검증 Recall:', np.mean(performance_metrics['recall']))
    print('## 평균 검증 F1:', np.mean(performance_metrics['f1']))

    ##추가한 부분
    print('## 평균 confusion_matrix:',  np.mean(performance_metrics['confusion_matrix'], axis=0).astype(int))
    # print('## 평균 early_stopping_epoch:', np.mean(performance_metrics['early_stopping_epoch']))
    ##추가한 부분

    df_performance = pd.DataFrame(model_performance)
    return df_performance



def calcul(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2)

def RegCV(model, feature, label, n_splits=5):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    model_performance = []
    
    Prd = []
    Y_t = []

    print(f"\n=== {model.__class__.__name__} ===")
    performance_metrics = {
        'Model': model.__class__.__name__, 
        
        'mse': [],
        'rmse': [],
        'rss': [],
        'mae': [],
        'r2': []
    }

    nets = [deepcopy(model) for _ in range(n_splits)]

    for i, (train_index, test_index) in enumerate(kfold.split(feature, label)):
        x_train, x_test = feature[train_index], feature[test_index]
        y_train, y_test = label[train_index], label[test_index]
        
        net = nets[i]
        net.fit(x_train, y_train)

        pred = net.predict(x_test)
        
        try:
            mse = mean_squared_error(y_test, pred)
            rmse = math.sqrt(mean_squared_error(y_test, pred))
            rss = calcul(y_test, pred)
            mae = mean_absolute_error(y_test, pred)
            r2 = r2_score(y_test, pred)
            Prd.append(pred)
            Y_t.append(y_test)

        except Exception as e:
            print("Error occurred:", str(e))
            mse, rss, mae, r2 = np.nan, np.nan, np.nan, np.nan

        
        print(f'\n#{i+1} 교차 검증 MSE: {mse}, RMSE: {rmse}, RSS: {rss}, MAE: {mae}, R^2: {r2}')

        performance_metrics['mse'].append(mse)
        performance_metrics['rmse'].append(rmse)
        performance_metrics['rss'].append(rss)
        performance_metrics['mae'].append(mae)
        performance_metrics['r2'].append(r2)
        

        # Prd = []
        # Y_t = []
        a = np.concatenate(Prd)
        b = np.concatenate(Y_t)
        plt.scatter(a,b)
        plt.show()
    
    avg_performance = {
        'Model': performance_metrics['Model'],
        'Avg MSE': np.mean(performance_metrics['mse']),
        'Avg RMSE': np.mean(performance_metrics['rmse']),
        'Avg RSS': np.mean(performance_metrics['rss']),
        'Avg MAE': np.mean(performance_metrics['mae']),
        'Avg R^2': np.mean(performance_metrics['r2'])
    }
    model_performance.append(avg_performance)
    # avg_performance = {}
    # model_performance.append(performance_metrics)

    print('\n## 평균 검증 MSE:', np.mean(performance_metrics['mse']))
    print('\n## 평균 검증 RMSE:', np.mean(performance_metrics['rmse']))
    print('## 평균 검증 RSS:', np.mean(performance_metrics['rss']))
    print('## 평균 검증 MAE:', np.mean(performance_metrics['mae']))
    print('## 평균 검증 R^2:', np.mean(performance_metrics['r2']))



    df_performance = pd.DataFrame(model_performance)
    return df_performance
