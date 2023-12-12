# est_wassup_02
ESTSoft Wassup 1기 첫번째 프로젝트

# 필수 참고! 강사님 모델 돌리시는건 seoik폴더에 이거돌리세요.ipynb 돌리시면 최종모델 성능평가 가능합니다.

2조 화이팅~~!!

1. 정서익
2. 이채욱
3. 임지혜
4. 강현준
5. 황소연
6. 임준수

* 프로젝트 주제 선정 및 데이터 수집 (11/27)
## 1. 프로젝트 목적
- 공공 데이터셋을 수집하여 주제를 선정하고 최적의 평가 결과를 도출하기 위한 학습 모델을 설계한다.

## 2. 프로젝트 주제 선정
### 전국구 자전거 사고 다발 지역 관련 정보 학습 모델 설계 후 모델 평가
- 자전거 사고 다발 지역 선정 조건
    1) 대상 사고 : 년도 별 발생한 가해 또는 피해 차종이 자전거인 교통사고
    2) 다발 지역 선정 조건 : 반경 300m 내 대상 사고 4건 이상 발생 지역

## 3. 데이터 수집
- 요구 데이터 : 전국구 자전거 사고 정보 (경기도, 서울시, 세종시, 등등..제주 포함 17개)
- 수집한 data set: 2017년~2022년에 발생한 자전거 교통사고 다발 지역에 포함된 사고의 개별 정보(사고번호, 사고일시, 요일, 시군구, 사고내용, 사망자수, 중상자수, 경상자수, 부상신고자수, 사고유형, 법규위반, 노면상태, 기상상태, 도로형태, 가해운전자 차종, 가해운전자 성별, 가해운전자 연령, 가해운전자 상해정도, 피해운전자 차종, 피해운전자 성별, 피해운전자 연령, 피해운전자 상해정도)
- 총 데이터량: 32,633

## EDA
1. 요약 통계 : describe, boxplot
2. 시각화 : seaborn (추가 예정)
3. 상관 관계 탐색 : correlation 그래프 그려보기
4. 이상치 탐색 : 박스 플롯으로 한 번 확인하기
5. 결측치 처리 : 이 데이터에는 크게 상관없음

### 기타불명 리스트 (처리해야 할 널값)
- 가해운전자 성별
- 가해운전 상해정도
- 피해운전자 차종
- 피해운전자 성별
- 피해운전자 연령
- 피해운전자 상해정도

### 상해 정도의 대략적 비율
- 경상 + 부상신고 : 15000
- 기타불명 +  상해없음 : 10000
- 중상 + 사망 : 5000
=> 비중 차이로 인한 고려 필요

### 데이터 불균형
- y값, 타겟의 불균형 맞추기
- 나머지 칼럼들도 데이터 분포가 균등할 필요는 없음

## 모델 공장
### 1.전처리 : data/base.csv 에서 칼럼을 제거하며 모델의 성능을 파악함
        사용법 : 
        prepro() 함수를 사용함
        prepro("파일경로", ["삭제할 칼럼", "삭제할 칼럼2"]) --> 삭제할 칼럼 여러개를 집어 넣어도 된다. 단, 존재해야 함
        prepro("파일경로") --> 삭제할 칼럼을 적지 않으면 삭제하지 않고 원핫인코딩한 pd를 리턴해준다.
        df = prepro("파일경로") --> df에 저장함

### 2.데이터 나누기 : 
        사용법 : 
        get() 함수를 사용하세요
        get("파일경로") --> 판다스 파일을 보내주면 알아서 train데이터 test데이터 validation데이터로 나눠준다. 
                            train:test=8:2 비율로 나눔
        X_trn, y_trn, X_tst, y_tst = get('파일경로')  --> 이런식으로 리턴이 총 4개 항목이기에 변수 4개에 넣어준다.

### 3.모델 여러개 만들기
        사용법 : 
        모델 : nn.py 
        config 파일 여러개 생성하여 각 파라미터 값을 다르게 설정한다.

# 커스텀모델 함수 사용하기

### 인공신경망 만드는 함수
- factiory_model( 트레이닝 데이터(X값=칼럼=피쳐=열) ,  타겟 데이터(맞춰야하는 데이터) , 하이퍼 파라미터 셋팅한 json파일의 이름)
-> EX :  factory_model(X_trn, y_trn,'config.json')

          리턴값(아웃풋)은 인공신경망 모델을 리턴함 학습도 가능함
          model = factory_model(X_trn, y_trn,'config.json')
          model.fit(X_trn, y_trn)
### 인공신경망 모델 여러개 만드는 함수
- create_models( 트레이닝 데이터(X값=칼럼=피쳐=열) ,  타겟 데이터(맞춰야하는 데이터), 하이퍼 파라미터 셋팅한 json파일의 이름 'configW.json' 무조건 W가 뒤에 붙어있어야함 , 몇개의 config.json파일을 넣을 것인지 )
-> EX :  create_models(X_trn, y_trn,'configW.json',2)

          리턴값(아웃풋)은 인공신경망 모델 여러개를 담은 리스트를 리턴함 인덱스로 꺼내서 사용 가능함
          models = create_models(X_trn, y_trn,'configW.json',2)
          model1 = models[0]
### 인공신경만 모델에 대한 교차검증 함수
- CV(모델(한개의 모델), 트레이닝 데이터(X값=칼럼=피쳐=열) ,  타겟 데이터(맞춰야하는 데이터), 5번 교차 검증을 할 것이고 기본값이기에 안적어도 됨)
-> EX :  CV(m, feature, label, n_splits=5)

          리턴값(아웃풋)은 판다스 데이터프레임이다 교차검증 후 5번 평균의 결과값을 가지고 있음 다음과 같이 프레임 확인도 가능함
          d = CV(m, feature, label, n_splits=5)  
          d.head()
### 인공신경만 모델 여러개에 대한 교차검증 함수
- models_cv(모델들(여러개의 모델), 트레이닝 데이터(X값=칼럼=피쳐=열) ,  타겟 데이터(맞춰야하는 데이터), 5번 교차 검증을 할 것이고 기본값이기에 안적어도 됨)
-> EX :  models_cv(models, feature, label, n_splits=5)

          리턴값(아웃풋)은 판다스 데이터프레임이다 여러 모델들을 순서대로 하나씩 5번 교차검증 후 데이터 평균 결과값을 한 행마다 스택처럼 쌓아서 리턴함
          d = models_cv(models, feature, label, n_splits=5) 
          d.head()


### 4. 모델 여러개 cross validation 교차검증 하기 
- HyperParameters/config.json 을 리스트 형식으로 만듬.
- layer, lr, epoch, feature 각각에 변화를 주고 그에 따른 결과를 관찰함
* config.json

        {
                "input_dim": "feature.shape[1]",
                "layers": [64, 256, 512, 3],
                "activations": ["leaky_relu","softplus","selu"],
                "dropout": 0.2,
                "lr": 0.0001,
                "epochs": 200,
                "batch_size": 256,
                "criterion" : "nn.CrossEntropyLoss()",
                "optimizer" : "Adam"
        }

### 5. 최종 모델 선정
- '사고 위험율을 예측하고 이를 대비하는 것' 에 중점을 두었기에 
Accuracy와 Precision(정확도와 정밀도)보단 F1_Score와 Recall(재현율)을 평가지표로 삼음. 
- 평가점수가 가장 높은 class 2의 6번 모델을 선정함.
class 2 recall : 0.4383057
class 2 f1_score : 0.3698523
 
### 6. PPT 작성 및 결과 발표
[PPT 발표 링크](https://www.canva.com/design/DAF2EUMGQJo/D8mbuwmtDY1TNUWMI_DxJA/edit?utm_content=DAF2EUMGQJo&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

