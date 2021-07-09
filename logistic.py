# 필요한 라이브러리 import
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import pandas as pd  

wine_data = datasets.load_wine()
""" 데이터 셋을 살펴보는 코드
print(wine_data.DESCR)
"""

# 입력 변수를 사용하기 편하게 pandas dataframe으로 변환
X = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)

# 목표 변수를 사용하기 편하게 pandas dataframe으로 변환
y = pd.DataFrame(wine_data.target, columns=['Y/N'])

# 코드를 쓰세요

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=5)
logistic_model=LogisticRegression(solver='saga',max_iter=7500)
logistic_model.fit(X_train,y_train)
y_test_predict=logistic_model.predict(X_test)

# 테스트 코드
score = logistic_model.score(X_test, y_test)
y_test_predict, score