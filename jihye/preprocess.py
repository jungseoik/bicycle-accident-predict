import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def get_X(df:pd.DataFrame, features:iter=['요일','사고내용','사망자수','중상자수','경상자수',
                                          '부상신고자수','사고유형','법규위반','노면상태',
                                          '기상상태','도로형태','가해운전자 차종','가해운전자 성별',
                                          '가해운전자 연령','피해운전자 차종','피해운전자 성별',
                                          '피해운전자 연령','피해운전자 상해정도','월',
                                          '도광역시','하루시간구분']):
  '''Make feature vectors from a DataFrame.

  Args:
      df: DataFrame
      features: selected columns
  '''
  df = df[features]
  object_columns = df.select_dtypes(include=['object']).columns
# 해당 열들에 대해서 원핫 인코딩 적용
  df = pd.get_dummies(df, columns=object_columns,dtype=int)

  # from https://www.kaggle.com/code/alexisbcook/titanic-tutorial
  return df.to_numpy(dtype=np.float32)

def get_y(df:pd.DataFrame):
  '''Make the target from a DataFrame.

  Args:
      df: DataFrame
  '''
  return df.타겟.to_numpy(dtype=np.float32)


def split(X,y,test_size=0.2, random_state=42) :
    

# Assuming you have your features (X) and labels (y)
# X and y should be NumPy arrays or Pandas DataFrames

# Split the data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
  return X_train, X_test, y_train, y_test
