# 필요한 라이브러리 import
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd  

# 당뇨병 데이터 갖고 오기
diabetes_dataset = datasets.load_diabetes()

# 입력 변수를 사용하기 편하게 pandas dataframe으로 변환
X = pd.DataFrame(diabetes_dataset.data, columns=diabetes_dataset.feature_names)

# 목표 변수를 사용하기 편하게 pandas dataframe으로 변환
y = pd.DataFrame(diabetes_dataset.target, columns=['diabetes'])

# train_test_split를 사용해서 주어진 데이터를 학습, 테스트 데이터로 나눈다
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=5)

linear_regression_model = LinearRegression()  # 선형 회귀 모델을 가지고 오고 
linear_regression_model.fit(X_train, y_train)  # 학습 데이터를 이용해서 모델을 학습 시킨다

y_test_predict = linear_regression_model.predict(X_test)  # 학습시킨 모델로 예측

# 평균 제곱 오차의 루트를 통해서 테스트 데이터에서의 모델 성능 판단
mse = mean_squared_error(y_test, y_test_predict)

mse ** 0.5

