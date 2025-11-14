import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score,recall_score, f1_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from A_catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
import holidays

#!pip install holidays

kr_holidays = holidays.KR()

df_raw = pd.read_csv("data_week2.csv",encoding="CP949")

df = df_raw.copy()

df.columns

df.columns = ['num', 'datetime', 'target', 'temp', 'wind', 'humid', 'rain', 'sunny', 'cooler', 'solar']
df["datetime"] = pd.to_datetime(df["datetime"])
df['month'] = df.datetime.dt.month                    # 월(숫자)
df['day'] = df.datetime.dt.day                        # 일(숫자)
df['hour'] = df.datetime.dt.hour                      # 시(숫자)
df['weekday'] = df.datetime.dt.weekday                # 요일(숫자)
df['holiday'] = df['datetime'].apply(lambda x: 1 if x in kr_holidays else 0)  # 공휴일
df['week'] = (df['datetime'] - df['datetime'].min()).dt.days // 7 + 1

# df["target"].describe()
# df["target"].hist()

# df["temp"].describe()
# df["temp"].hist()

# df["wind"].describe()
# df["wind"].hist()

# df["humid"].describe()
# df["humid"].hist()

# df["rain"].describe()
# df["rain"].hist()

# 전처리
df['log_target'] = np.log(df['target'] + 1)

# 각 건물의 전력 소비량 평균 계산 (로그 변환된 값)
building_avg_consumption_log = df.groupby('num')['log_target'].mean()

# 로그 변환된 확률 밀도 함수(PDF) 그리기
plt.figure(figsize=(10, 6))
sns.kdeplot(building_avg_consumption_log, color='blue', shade=True)

# 그래프 제목 및 레이블 설정
plt.title('Probability Density Function of Log-Transformed Average Power Consumption', fontsize=16)
plt.xlabel('Log(Average Power Consumption)', fontsize=12)
plt.ylabel('Density', fontsize=12)

# 그래프 출력
plt.grid(True)
plt.show()


train = []
valid = []
for num, group in df.groupby('num'):
    train.append(group.iloc[:len(group)-7*24])  
    valid.append(group.iloc[len(group)-7*24:]) 


train_df = pd.concat(train)
train_x = train_df.drop("target",axis=1)
train_y = train_df["target"] 

valid_df = pd.concat(valid)
valid_x = valid_df.drop("target",axis=1)
valid_y = valid_df["target"] 

# 날짜형 컬럼(datetime) 제거
train_x = train_x.drop(columns=['datetime'], errors='ignore')
valid_x = valid_x.drop(columns=['datetime'], errors='ignore')


#####lgbm
lgb_model = LGBMRegressor(random_state=42)
lgb_model.fit(train_x, train_y)

lgb_pred = lgb_model.predict(valid_x)
lgb_mse = mean_squared_error(valid_y, lgb_pred)
print(lgb_mse)


#####catboost
cat_model = CatBoostRegressor(random_state=42, verbose=0)  # verbose=0: 학습 로그 출력하지 않음
cat_model.fit(train_x, train_y)

cat_pred = cat_model.predict(valid_x)
cat_mse = mean_squared_error(valid_y, cat_pred)
print(cat_mse)


#fig = plt.figure(figsize = (15, 40))
#for num in range(1,61):
#    ax = plt.subplot(12, 5, num)
#    energy = np.log(1 + df.loc[df.num == num, 'target'].values)
#    mean = energy.mean().round(3)
#    std = energy.std().round(3)
#    plt.hist(energy, alpha = 0.7, bins = 50)
#    plt.title(f'building{num}')
#    plt.xticks([])
#    plt.yticks([])
#    plt.xlabel('')
#    plt.ylabel('')
#    
#sns.lineplot(data=df,x="temperature",y="target")
#sns.lineplot(data=df,x="temperature",y="target",hue="num")