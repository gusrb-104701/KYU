# -*- coding: utf-8 -*-
"""
Created on Wed May 21 16:22:54 2025

@author: gusrb
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import shap
import matplotlib.pyplot as plt

# 1. 데이터 불러오기
df = pd.read_csv(r"C:\Users\gusrb\Desktop\가상공정 DATA.csv")

# 2. 불필요한 열 제거
df = df.loc[:, ~df.columns.str.contains('Unnamed')]

# 3. Line Ratio 계산
df['Ratio_400_600'] = df['OES_400nm'] / df['OES_600nm']
df['Ratio_280_400'] = df['OES_280nm'] / df['OES_400nm']

# 4. CRM 기반 Te, Ne 생성
df['Te_CRM'] = 0.05 * df['Ratio_400_600'] ** 2 + 1.2
df['Ne_CRM'] = 1e16 * np.exp(-df['Ratio_280_400'])

# 5. 입력변수(X), 라벨(y) 설정
feature_cols = ['OES_280nm', 'OES_400nm', 'OES_600nm', 'Power', 'Pressure', 'GasFlow']
X = df[feature_cols]
y = df['Te_CRM']  # ← Te 예측만 실습 (Ne도 나중에 가능!)

# 6. 학습/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. 모델 학습
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 8. 예측 및 평가
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# 9. 결과 시각화
plt.figure(figsize=(8, 4))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel("True Te")
plt.ylabel("Predicted Te")
plt.title("Te Prediction Result")
plt.grid(True)
plt.tight_layout()
plt.show()

# 10. SHAP 분석 (CPU-safe로)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title("SHAP Feature Importance (Te 예측)")
plt.show()
