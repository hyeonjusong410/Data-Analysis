import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_csv('data_week4.csv', encoding='cp949')

df.columns
# 'line', 'name', 'mold_name', 'emergency_stop' 드랍
df = df.drop(['line', 'name', 'mold_name', 'emergency_stop', 'Unnamed: 0'], axis=1)

# low_section_speed 두 변수 선형성
plt.figure(figsize=(8, 6))
sns.regplot(x='cast_pressure', y='low_section_speed', data=df, scatter_kws={'s':50}, line_kws={'color':'red'})
plt.title('Casting Pressure vs  low_section_speed')
plt.xlabel('Casting Pressure (Pa)')
plt.ylabel('Physical Strength (N)')
plt.grid(True)

#=> 이상치?
# 숫자형 데이터만 대상으로 합계를 구하는 코드
filtered_sum = df[df['low_section_speed'] >= 60000]
df = df[df['low_section_speed'] < 60000]


# high_section_speed 두 변수 선형성
plt.figure(figsize=(8, 6))
sns.regplot(x='cast_pressure', y='high_section_speed', data=df, scatter_kws={'s':50}, line_kws={'color':'red'})
plt.title('Casting Pressure vs  high_section_speed')
plt.xlabel('Casting Pressure (Pa)')
plt.ylabel('Physical Strength (N)')
plt.grid(True)

# 박스플랏 그리기
plt.figure(figsize=(12, 6))

# Casting Pressure 박스플랏
plt.subplot(1, 2, 1)
sns.boxplot(x=df['low_section_speed'])
plt.title('Box Plot of low_section_speed')
plt.xlabel('low_section_speed')

# Physical Strength 박스플랏
plt.subplot(1, 2, 2)
sns.boxplot(x=df['high_section_speed'])
plt.title('Box Plot of high_section_speed')
plt.xlabel('high_section_speed')

plt.tight_layout()
plt.show()




