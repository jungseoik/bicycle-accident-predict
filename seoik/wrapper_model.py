import json
import torch
from torch import nn, optim
from nn import DynamicANN, DynamicANNWrapper
from sklearn.model_selection import train_test_split
import os
# config_file ,
# 현재 디렉토리의 하위 디렉토리에 있는 config.json 파일을 읽기
# 현재 작업 디렉토리 가져오기


def wrapper_model(feature, label ,testF ,jsonstr):
    # 설정 파일 로드
    current_dir = os.getcwd()
    # 파일 경로 결합
    file_path = os.path.join(current_dir, jsonstr)
    with open(file_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    code_str = config['input_dim']   
    input_dim = eval(code_str)
    # 설정 추출
    
    layers = config['layers']
    activations = config['activations']
    dropout = config['dropout']
    lr = config['lr']
    epochs = config['epochs']
    batch_size = config['batch_size']
    criterion = config['criterion']
    ##############################################위에가 json에서 데이터 받는 부분

    # 모델 생성
    model_wrapper = DynamicANNWrapper(input_dim, layers, activations, dropout ,epochs, batch_size, criterion , lr)

    model_wrapper.fit(feature, label)

    models = []
    models.append(model_wrapper)

    return models





def create_models(feature, label, testF, config_file, num_models):
    models = []
    for i in range(1, num_models+1):
        # config_file 문자열에서 "W1", "W2", "W3"을 숫자로 대체
        print("아오 이거 잘되는거 맞아?")
        config_file = config_file.replace("W1", "W" + str(i))
        print(config_file)

        model_variable_name = "model" + str(i)  # 동적으로 변수명 생성
        locals()[model_variable_name] = wrapper_model(feature, label, testF, config_file)
        models.append(locals()[model_variable_name])

        # model = wrapper_model(feature, label, testF, config_file)
        # models.append(model)
    return models

