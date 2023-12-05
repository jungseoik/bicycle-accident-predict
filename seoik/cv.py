from tqdm.auto import tqdm
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import mean_squared_error
import numpy as np
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError
from typing import Optional, List
import torchmetrics
from copy import deepcopy
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import torch

def CV(model, feature, label, n_splits=5):
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
