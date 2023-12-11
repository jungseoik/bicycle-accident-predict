from get import *
from cv import *
from nn import * 
from factory_model import *
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Load your training and testing data
# Replace the following lines with your actual data loading logic

num_models = 1
batch_size = 256

X_trn, y_trn, X_tst, y_tst = get('data/f5.csv')
#config_file = "your_config_file.json"  # Replace with your actual config file
model = factory_model(X_tst, y_tst, "config.json")
#X_train, y_train = load_training_data()
#X_test, y_test = load_testing_data()

# Create an instance of your model
#model = SimpleModel(input_size, hidden_size, num_classes)


model.fit(X_trn, y_trn)

b = 9
eval_results = []

test_dataset = TensorDataset(torch.tensor(X_tst, dtype=torch.float32), torch.tensor(y_tst, dtype=torch.long))
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
eval_loss, eval_metrics_df = evaluate(model, nn.CrossEntropyLoss(), test_loader, 'cuda' if torch.cuda.is_available() else 'cpu')

    # Print and save evaluation metrics
print(f"Model {b} evaluation results:")
print(eval_metrics_df)

eval_metrics_df.to_csv(f'model_{b}_evaluation_metrics.csv', index=False)
print(f"Model {b} evaluation metrics saved to 'model_{b}_evaluation_metrics.csv'")

eval_results.append(eval_loss)

print("Overall evaluation results:")
for i, result in enumerate(eval_results, start=1):
    print(f"Model {b}: Loss = {result:.4f}")

#모델 저장하기
model_path = f'model_{b}.pth'
torch.save(model.model.state_dict(), model_path)
print(f"Model {b} saved to '{model_path}'")