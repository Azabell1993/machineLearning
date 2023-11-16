import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# 데이터 생성
n_samples = 1000
X, y = make_regression(n_samples=n_samples, n_features=5, noise=0.1, random_state=42)

# 데이터프레임 생성
columns = ['Temperature', 'Humidity', 'WindSpeed', 'Rainfall', 'Holiday', 'BikeRentals']
df_bike = pd.DataFrame(np.column_stack([X, y]), columns=columns)

# CSV 파일로 저장
df_bike.to_csv('bike_rentals.csv', index=False)

# 데이터 로드
df_bike = pd.read_csv('bike_rentals.csv')

# 데이터 파악하기
print(df_bike.dtypes)

# 학습 및 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(df_bike.drop('BikeRentals', axis=1), df_bike['BikeRentals'].values, test_size=0.2, random_state=42)

# 랜덤 포레스트 모델 생성
model = RandomForestRegressor(n_estimators=100, random_state=42)

# 모델 학습
model.fit(X_train, y_train)

# 테스트 데이터로 예측
predictions = model.predict(X_test)

# 결과 시각화
plt.scatter(y_test, predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.show()

