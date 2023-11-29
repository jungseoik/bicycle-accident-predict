

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
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def CV(models, feature, label, n_splits=5):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    model_performance = []

    for model in models:
        print(f"\n=== {model.__class__.__name__} ===")
        performance_metrics = {
            'Model': model.__class__.__name__, 
            
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }

        nets = [deepcopy(model) for _ in range(n_splits)]

        for i, (train_index, test_index) in enumerate(kfold.split(feature, label)):
            x_train, x_test = feature[train_index], feature[test_index]
            y_train, y_test = label[train_index], label[test_index]

            net = nets[i]
            net.fit(x_train, y_train)

            pred = net.predict(x_test)
            
            try:
                accuracy = accuracy_score(y_test, pred)
                precision = precision_score(y_test, pred, average='macro')
                recall = recall_score(y_test, pred, average='macro')
                f1 = f1_score(y_test, pred, average='macro')
            except Exception as e:
                print("Error occurred:", str(e))
                accuracy, precision, recall, f1 = np.nan, np.nan, np.nan, np.nan

            print(f'\n#{i+1} 교차 검증 Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}')
            print('#{0} 검증 세트 인덱스 : {1}'.format(i+1, test_index))

            performance_metrics['accuracy'].append(accuracy)
            performance_metrics['precision'].append(precision)
            performance_metrics['recall'].append(recall)
            performance_metrics['f1'].append(f1)
            
        avg_performance = {
            'Model': performance_metrics['Model'],
            'Avg Accuracy': np.mean(performance_metrics['accuracy']),
            'Avg Precision': np.mean(performance_metrics['precision']),
            'Avg Recall': np.mean(performance_metrics['recall']),
            'Avg F1': np.mean(performance_metrics['f1'])
        }
        model_performance.append(avg_performance)

        print('\n## 평균 검증 Accuracy:', np.mean(performance_metrics['accuracy']))
        print('## 평균 검증 Precision:', np.mean(performance_metrics['precision']))
        print('## 평균 검증 Recall:', np.mean(performance_metrics['recall']))
        print('## 평균 검증 F1:', np.mean(performance_metrics['f1']))

    df_performance = pd.DataFrame(model_performance)
    return df_performance