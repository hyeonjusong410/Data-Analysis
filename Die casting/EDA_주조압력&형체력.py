import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# !pip install matplotlib
# !pip install seaborn

df= pd.read_csv('data_week4.csv', encoding='cp949')

df.columns
# 'line', 'name', 'mold_name', 'emergency_stop' 드랍
df = df.drop(['line', 'name', 'mold_name', 'emergency_stop', 'Unnamed: 0'], axis=1)

# 두 변수 선형성
plt.figure(figsize=(8, 6))
sns.regplot(x='cast_pressure', y='physical_strength', data=df, scatter_kws={'s':50}, line_kws={'color':'red'})
plt.title('Casting Pressure vs Physical Strength')
plt.xlabel('Casting Pressure (Pa)')
plt.ylabel('Physical Strength (N)')
plt.grid(True)
# 그래프 출력
plt.show()
# =>  이상치?

# 이상치 제거
df = df[df['physical_strength'] < 60000]

df[df['physical_strength'] >= 60000]

# 박스플랏 그리기
plt.figure(figsize=(12, 6))

# Casting Pressure 박스플랏
plt.subplot(1, 2, 1)
sns.boxplot(x=df['cast_pressure'])
plt.title('Box Plot of Casting Pressure')
plt.xlabel('Casting Pressure (Pa)')

# Physical Strength 박스플랏
plt.subplot(1, 2, 2)
sns.boxplot(x=df['physical_strength'])
plt.title('Box Plot of Physical Strength')
plt.xlabel('Physical Strength (N)')

plt.tight_layout()
plt.show()

# 두 변수 간의 상관계수 계산
correlation = df['cast_pressure'].corr(df['physical_strength'])
print(f"두 변수 간의 상관계수: {correlation}")

# 히트맵
plt.figure(figsize=(8, 6))
sns.kdeplot(x=df['cast_pressure'], y=df['physical_strength'], cmap="Reds", fill=True)
plt.title('Density Plot: Casting Pressure vs Physical Strength')
plt.xlabel('Casting Pressure (Pa)')
plt.ylabel('Physical Strength (N)')
plt.grid(True)
plt.show()


