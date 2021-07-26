# 필요한 도구들을 가지고 오는 코드
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

# 유저 평점 + 영화 속성 데이터 경로 정의
MOVIE_DATA_PATH = './data/movie_rating.csv'

# pandas로 데이터 불러 오기
movie_rating_df = pd.read_csv(MOVIE_DATA_PATH)

features =['romance', 'action', 'comedy', 'heart-warming'] # 사용할 속성들 이름

# 입력 변수와 목표 변수 나누기
X = movie_rating_df[features]
y = movie_rating_df[['rating']]

# 입력 변수와 목표 변수들을 각각의 training/test 셋으로 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
# 코드를 쓰세요.
model=LinearRegression()
model.fit(X_train,y_train)
y_test_predict=model.predict(X_test)
# 실행 코드
y_test_predict
