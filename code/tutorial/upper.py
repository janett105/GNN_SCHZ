import numpy as np

# 대칭 행렬 예시
symmetric_matrix = np.array([
    [1, 2, 3],
    [2, 4, 5],
    [3, 5, 6]
])

# 대칭 행렬의 upper triangle 부분을 1차원 배열로 변환
# k=1 옵션은 대각선을 제외한 upper triangle을 선택하기 위함
upper_triangle = symmetric_matrix[np.triu_indices_from(symmetric_matrix, k=1)]

print(upper_triangle)