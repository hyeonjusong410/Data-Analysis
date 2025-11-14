import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("1주_실습데이터.csv")
df.info()
df.head()

mpl.rc('font', size=15)
sns.displot(df['Y'])

sum(df["X4"] == 0.015348)

# 통계적 검정
from sklearn.feature_selection import f_classif, chi2
from sklearn.preprocessing import MinMaxScaler


# 독립 변수(X)와 타겟(Y) 분리
X = df.drop('Y', axis=1)
Y = df['Y']

# 스케일링 (chi2가 양수 값만 다루기 때문에 MinMaxScaler를 사용)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ANOVA F-test 수행 (f_classif)
f_test, f_p_values = f_classif(X_scaled, Y)

# Chi-square test 수행 (chi2)
chi2_test, chi2_p_values = chi2(X_scaled, Y)

# 결과를 DataFrame으로 정리
statistical_test_results = pd.DataFrame({
    'Variable': X.columns,
    'F-Test Score': f_test,
    'F-Test p-value': f_p_values,
    'Chi2 Score': chi2_test,
    'Chi2 p-value': chi2_p_values
})

# 결과 확인
statistical_test_results

f_top = statistical_test_results['F-Test Score'].sort_values(ascending=False).head(5)
f_bottom = statistical_test_results['F-Test Score'].sort_values(ascending=False).tail(5)
f_p = statistical_test_results[statistical_test_results['F-Test p-value'] > 0.05]
C_top = statistical_test_results['Chi2 Score'].sort_values(ascending=False).head(5)
C_bottom = statistical_test_results['Chi2 Score'].sort_values(ascending=False).tail(5)
C_p = statistical_test_results[statistical_test_results['Chi2 p-value'] > 0.05]

from collections import Counter

# 다수 클래스 플롯
plt.scatter(X_scaled[Y == 0][:, 0], X_scaled[Y == 0][:, 1], label="Majority Class", c='blue', marker='o', s=100)

# 소수 클래스 플롯
plt.scatter(X_scaled[Y == 1][:, 0], X_scaled[Y == 1][:, 1], label="Minority Class", c='red', marker='x', s=100)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Imbalanced Data Distribution")
plt.legend()
plt.show()

# 클래스 분포 확인
print("클래스 분포:", Counter(Y))