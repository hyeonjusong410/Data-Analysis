import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import f1_score, classification_report
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from collections import Counter


# 데이터 로드
df = pd.read_csv("1주_실습데이터.csv")

# feature engineering
df = df.drop(columns=['X4', 'X13', 'X18', 'X19', 'X20'])

X = df.drop("Y", axis=1)
y = df['Y']

# train/test 셋 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 칼럼별 왜도 계산--------------------------------------------------
# skewness_values = X_train.apply(lambda x: x.skew())

# # 결과를 데이터프레임으로 변환
# skewness_table = pd.DataFrame({
#     'Feature': skewness_values.index,
#     'Skewness': skewness_values.values
# })

# # 왜도 테이블 출력
# print(skewness_table)

# # 로그/제곱 변환을 적용할 변수 선택
# log_transform_features = skewness_table[skewness_table['Skewness'] > 2]['Feature'].tolist()
# square_transform_features = skewness_table[skewness_table['Skewness'] < -2]['Feature'].tolist()

# # 변환 적용
# for feature in log_transform_features:
#     X_train[feature] = np.log1p(X_train[feature])  # log(1 + x)
#     X_test[feature] = np.log1p(X_test[feature])    # test set에도 동일한 변환 적용

# for feature in square_transform_features:
#     X_train[feature] = np.square(X_train[feature])
#     X_test[feature] = np.square(X_test[feature])

# StandardScaler 적용
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# 스케일된 데이터를 DataFrame으로 변환
# X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
# X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
#-----------------------------------------------------------------------

# Polynomial Features 적용 
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# 스케일된 데이터를 DataFrame으로 변환
X_train_poly = pd.DataFrame(X_train_poly, columns=poly.get_feature_names_out(X.columns))
X_test_poly = pd.DataFrame(X_test_poly, columns=poly.get_feature_names_out(X.columns))

# SMOTE 적용
smote = SMOTE(sampling_strategy=0.3, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_poly, y_train)

# SMOTE 적용 후 클래스 분포 확인
print(f"Resampled dataset shape: {Counter(y_train_resampled)}")

# LightGBM 모델 초기화
model = lgb.LGBMClassifier(
    learning_rate=0.1789067697356303, 
    max_depth=-1, 
    n_estimators=387, 
    num_leaves=130
)

# 모델 학습
model.fit(
    X_train_resampled, y_train_resampled
)

# 예측
y_pred = model.predict(X_test_poly)
y_pred_proba = model.predict_proba(X_test_poly)

# 평가지표 계산 함수 작성
from sklearn.metrics import f1_score, roc_auc_score
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import roc_curve, auc

#!pip install imblearn
def Eval(y_true, y_pred, y_pred_proba=None):
    # F1 score 계산
    f1 = f1_score(np.round(y_true), np.round(y_pred), average='weighted')
    
    # Geometric Mean (G-Mean) 계산
    gmean = geometric_mean_score(np.round(y_true), np.round(y_pred), average='weighted')
    
    
    # AUC 계산 
    auc_score = roc_auc_score(y_true, y_pred_proba[:, 1]) if y_pred_proba is not None else None
      
    return {
        "F1 Score": f1,
        "G-Mean": gmean,
        "AUC": auc_score
    }

 # 성능 평가
results = Eval(y_test, y_pred, y_pred_proba)

# 결과 출력
for metric, value in results.items():
    print(f"{metric}: {value}")

# 혼동행렬 시각화
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# 혼동 행렬 계산
conf_matrix = confusion_matrix(y_test, y_pred)

# 혼동 행렬 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greys', cbar=False,
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
