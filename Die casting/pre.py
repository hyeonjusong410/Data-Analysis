import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score
from sklearn.ensemble import RandomForestClassifier

# 데이터 불러오기
df_raw = pd.read_csv("data_week4.csv", encoding="CP949")
df = df_raw.copy()

# 시운전 "D" 제거
df.drop(df[df.tryshot_signal == "D"].index, inplace=True)

# date, time과 registration_time이 같은가? 같다 -> 제거
df["date_time"] = pd.to_datetime(df["time"] + " " + df["date"])
df["registration_time"] = pd.to_datetime(df["registration_time"])
sum(df["registration_time"] != df["date_time"])

# 시간 변수 생성
df["day"] = df["registration_time"].dt.day
df["hour"] = df["registration_time"].dt.hour
df["weekday"] = df["registration_time"].dt.day_of_week

# 필요 없는 칼럼 제거
df.drop(["Unnamed: 0", "line", "name", "mold_name", "emergency_stop", "tryshot_signal", 
         "registration_time", "count", "time", "date", "date_time"], axis=1, inplace=True)
df.drop(index=19327, inplace=True)

# 결측치 채우기
df["working"] = df["working"].fillna("가동")
df["heating_furnace"] = df["heating_furnace"].fillna("C")
df['molten_temp'] = df['molten_temp'].fillna(df['molten_temp'].median())
df["upper_mold_temp3"].fillna(1449.0, inplace=True)
df["lower_mold_temp3"].fillna(1449.0, inplace=True)

# 1. Area 계산 (형체력 / 주조압력)
df['Area'] = df['physical_strength'] / df['cast_pressure']

# 2. PSI 계산 (Force / Area)
# Force를 형체력으로 설정한 상태에서, Area를 사용하여 PSI 계산
df['PSI'] = df['physical_strength'] / df['Area']

df.fillna(0,inplace=True)

object_columns = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=object_columns, drop_first=True)

# X, Y 분리
X = df.drop('passorfail', axis=1)
Y = df['passorfail']

# 데이터셋 분할
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

index_to_drop = X_train[X_train['low_section_speed'] > 60000].index
X_train.drop(index_to_drop, inplace=True)
Y_train.drop(index_to_drop, inplace=True)

index_to_drop = X_train[X_train['Coolant_temperature'] > 1400].index
X_train.drop(index_to_drop, inplace=True)
Y_train.drop(index_to_drop, inplace=True)

index_to_drop = X_train[X_train['physical_strength'] > 60000].index
X_train.drop(index_to_drop, inplace=True)
Y_train.drop(index_to_drop, inplace=True)

index_to_drop = X_train[X_train['upper_mold_temp2'] > 4000].index
X_train.drop(index_to_drop, inplace=True)
Y_train.drop(index_to_drop, inplace=True)

# 결측치가 있는 부분만 KNNImputer로 채우기
imputer = KNNImputer(n_neighbors=3)

molten_temp_na_indices = X_train['molten_temp'].isna()

df_filled_train = imputer.fit_transform(X_train[['molten_temp']])

X_train.loc[molten_temp_na_indices, 'molten_temp'] = df_filled_train[molten_temp_na_indices]

# 테스트 데이터에서도 동일하게 적용
molten_temp_na_indices_test = X_test['molten_temp'].isna()
df_filled_test = imputer.transform(X_test[['molten_temp']])
X_test.loc[molten_temp_na_indices_test, 'molten_temp'] = df_filled_test[molten_temp_na_indices_test]

# 파생 변수 생성 함수
def derived_var(df):
    df['molten_temp'] = df['molten_temp'].fillna(df['molten_temp'].median())

    df['molten_temp_diff'] = df['molten_temp'].diff()

    df['pressure_temp_interaction'] = df['molten_temp'] * df['EMS_operation_time']
    
    df['pressure_speed_ratio'] = df['EMS_operation_time'] / df['molten_temp']
    df['speed_change_rate'] = df['EMS_operation_time'].pct_change()
    df['speed_change_rate'].fillna(0, inplace=True)

    df['EMS_operation_time'].fillna(0, inplace=True)

    df['molten_temp_std'] = df['molten_temp'].expanding().std()
    df['molten_temp_std'].fillna(0, inplace=True)

    df["mean_temp_lower"] = (df['lower_mold_temp3'] + df['lower_mold_temp2'] + df['lower_mold_temp1']) / 3
    df["mean_temp_upper"] = (df['upper_mold_temp3'] + df['upper_mold_temp2'] + df['upper_mold_temp1']) / 3
    df["mean_temp_total"] = (df["mean_temp_lower"] + df["mean_temp_upper"]) / 2

    df["mean_temp_lower_diff"] = df["mean_temp_lower"].diff()
    df['mean_temp_lower_diff'].fillna(0, inplace=True)

    df["mean_temp_upper_diff"] = df["mean_temp_upper"].diff()
    df['mean_temp_upper_diff'].fillna(0, inplace=True)

    df["mean_temp_total_diff"] = df["mean_temp_total"].diff()
    df['mean_temp_total_diff'].fillna(0, inplace=True)
    
    df["hour_7"] = (df["hour"] == 7)

    df['pressure_mold_temp_interaction'] = df['EMS_operation_time'] * df['mean_temp_total_diff']

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    
    return df

# 파생 변수 생성
X_train = derived_var(X_train)
X_test = derived_var(X_test)
Y_test
X_test

pd.DataFrame(X_test) 

#---------------------------------------------------------------------------
# 의사결정나무 모델 생성 및 학습
decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X_train, Y_train)

# 테스트 데이터 예측 및 평가
y_pred_dt = decision_tree_model.predict(X_test)
recall_dt = recall_score(Y_test, y_pred_dt, average='binary') # 
accuracy_dt = accuracy_score(Y_test, y_pred_dt)

# 모델 평가 결과 출력
print(f'Decision Tree Accuracy: {accuracy_dt:.4f}')
print(f'Decision Tree Recall: {recall_dt:.4f}')

# 랜덤 포레스트 모델 생성 및 학습
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, Y_train)

#----------------------------------------------------------------------------
# 변수 중요도 시각화 (랜덤 포레스트)
importances_rf = rf_model.feature_importances_
feature_names = X_train.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=importances_rf, y=feature_names)
plt.title('Random Forest Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()

# 변수 중요도 시각화 (의사결정나무)
importances_dt = decision_tree_model.feature_importances_

plt.figure(figsize=(10, 6))
sns.barplot(x=importances_dt, y=feature_names)
plt.title('Decision Tree Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()