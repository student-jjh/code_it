import numpy as np
from math import sqrt


def distance(user_1, user_2):
    """유클리드 거리를 계산해주는 함수"""
    # 코드를 쓰세요 
    array_dif = user_2-user_1
    array_dif = array_dif * array_dif
    for_return = np.sum(array_dif)
    return sqrt(for_return)

# 실행 코드
user_1 = np.array([0, 1, 2, 3, 4, 5])
user_2 = np.array([0, 1, 4, 6, 1, 4])

distance(user_1, user_2)
