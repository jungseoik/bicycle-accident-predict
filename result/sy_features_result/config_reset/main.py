from get import get,prepro
from factory_model import factory_model, create_models , models_cv
from cv import CV

df = prepro("data/base.csv",['사망자수', '중상자수', '경상자수', '부상신고자수', '사고내용'])
df.info()
df.to_csv("data/f6.csv")
# df.to_csv("data/prepro_so_5.csv")
X_trn, y_trn, X_tst, y_tst = get('data/f6.csv')


models = create_models(X_trn,y_trn,'configW.json',7)

df = models_cv(models,X_trn,y_trn,n_splits=5)
df.head()
df.to_csv('result/sy_3_cf7.csv')

# 'Confusion_Matrix' 칼럼 출력
confusion_matrices = df['Avg Confusion Matrix']

# apply 함수를 사용하여 리스트의 내용을 출력
print(f"Confusion Matrix:")
for row in confusion_matrices:
    print(row)
