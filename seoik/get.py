import pandas as pd
from sklearn.model_selection import train_test_split

def get(filepath):
    df = pd.read_csv(filepath)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=2023)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=2023)

    X_trn = train_df.drop('타겟',axis=1).values
    y_trn = train_df['타겟'].values
    X_tst = test_df.drop('타겟',axis=1).values
    y_tst = test_df['타겟'].values
    X_val = val_df.drop('타겟',axis=1).values
    y_val = val_df['타겟'].values   

    return X_trn, y_trn, X_tst, y_tst, X_val, y_val

# 사용 예시
# X_trn, y_trn, X_tst, y_tst, X_val, y_val = get('prepro_jung.csv')

def prepro(filepath, list=None):
    df = pd.read_csv(filepath)
    df = df.drop(columns=list, errors='ignore')
    # object 타입인 열들을 선택
    object_columns = df.select_dtypes(include=['object']).columns
    # 해당 열들에 대해서 원핫 인코딩 적용
    df = pd.get_dummies(df, columns=object_columns,dtype=int)
    return df