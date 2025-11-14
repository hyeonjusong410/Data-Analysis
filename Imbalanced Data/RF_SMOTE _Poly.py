import pandas as pd
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from imblearn.over_sampling import SMOTE
from collections import Counter


# 데이터 불러오기
df = pd.read_csv("1주_실습데이터.csv")

# 불필요한 열 제거
df = df.drop(columns=['X4', 'X13', 'X18', 'X19', 'X20'])

# 독립변수(X)와 종속변수(y) 분리
X = df.drop("Y", axis=1)
y = df['Y']

# train/test 셋 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Polynomial Features 적용 (차수는 2로 설정)
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

#-------------------------------------------------------------------
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
#---------------------------------------------------------------------
# RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

# Random Forest 모델 생성
model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train_poly, y_train)

# 예측
y_pred = model.predict(X_test_poly)
y_pred_proba = model.predict_proba(X_test_poly)

# 성능 평가
results = Eval(y_test, y_pred, y_pred_proba)

# 결과 출력
for metric, value in results.items():
    print(f"{metric}: {value}")

