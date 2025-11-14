import numpy as np
import pandas as pd

df= pd.read_csv('data_week4.csv', encoding='cp949')

df.info()

df.isnull().sum()

df.shape
df.columns
df.head()

for column in df.columns:
    unique_values = df[column].nunique()
    print(f"'{column}' 변수의 유니크 값 개수: {unique_values}")
    
# 유니크 값이 1인 변수(열)들만 추출
unique_one_columns = [col for col in df.columns if df[col].nunique() == 1]

print(f"유니크 값이 1인 변수: {unique_one_columns}")

df[['line', 'name', 'mold_name', 'emergency_stop', 'tryshot_signal']].isnull().sum()


# 유니크 값이 2인 변수(열)들만 추출
unique_one_columns = [col for col in df.columns if df[col].nunique() == 2]

print(f"유니크 값이 2인 변수: {unique_one_columns}")
df[['working', 'passorfail', 'heating_furnace']].isnull().sum()
df['working'].unique()
df['passorfail'].unique()
df['heating_furnace'].unique()

# 범주형 컬럼만 추출
categorical_columns = df.select_dtypes(include=['object', 'category']).columns

print(f"범주형 컬럼: {list(categorical_columns)}")

# 'line', 'name', 'mold_name' 드랍
df = df.drop(['line', 'name', 'mold_name'], axis=1)

df['working'].unique()
(df['working'] == '가동').sum()
(df['working'] == '정지').sum()
df['working'].isnull().sum()

import matplotlib.pyplot as plt

# 수치형 변수만 선택
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

# 히스토그램 그리기
def plot_histograms(df, columns):
    df[columns].hist(figsize=(16, 12), bins=30, layout=(int(len(columns)/3)+1, 3))
    plt.tight_layout()
    plt.show()

# 히스토그램 그리기 호출
plot_histograms(df, numeric_columns)

df['time'].unique().shape # 85일
df['date'].head

df['mold_code'].dtype
df['mold_code'].unique()


# registration_time 열을 datetime 형식으로 변환
df['registration_time'] = pd.to_datetime(df['registration_time'])

# 날짜와 시간을 분리하여 새로운 열 추가
df['Date'] = df['registration_time'].dt.date
df['Hour'] = df['registration_time'].dt.hour

# 사탕 신호는 tryshot_signal로 가정하고, 그 값이 존재하는 행만 선택
df_clean = df.dropna(subset=['tryshot_signal'])

# 'Date' 열을 기준으로 하루씩 데이터를 그룹화
daily_groups = list(df_clean.groupby('Date'))

# 플롯을 5일 단위로 그리기
for start in range(0, len(daily_groups), 5):
    fig, axes = plt.subplots(nrows=min(5, len(daily_groups)-start), figsize=(10, 6 * 5))
    
    # 하루씩 끊어서 24시간을 x축으로 하고 y축은 사탕 신호로 하는 플랏 그리기
    for i, (day, group) in enumerate(daily_groups[start:start+5]):
        ax = axes[i]
        ax.plot(group['Hour'], group['tryshot_signal'], marker='o')
        ax.set_title(f'Tryshot Signal for {day}')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Tryshot Signal')
        ax.grid(True)

    plt.tight_layout()
    plt.show()

#0, 1 
failures_count = df['passorfail'].value_counts()

# 0과 1의 비율을 계산
failures_ratio = df['passorfail'].value_counts(normalize=True) * 100

# 결과 출력
print("Pass or Fail Count:")
print(failures_count)

print("\nPass or Fail Ratio (%):")
print(failures_ratio)