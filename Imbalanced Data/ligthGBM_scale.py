import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import lightgbm as lgb
from collections import Counter
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import roc_auc_score

# 데이터 로드
df = pd.read_csv("1주_실습데이터.csv")

# feature engineering
df = df.drop(columns=['X4', 'X13', 'X18', 'X19', 'X20'])

X = df.drop("Y", axis=1)
y = df['Y']

# train/test 셋 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 클래스 비율 계산 (소수 클래스의 비율에 따라 scale_pos_weight 설정)
class_counts = Counter(y_train)
scale_pos_weight = class_counts[0] / class_counts[1]

# LightGBM 모델 초기화
model = lgb.LGBMClassifier(
    learning_rate=0.1789067697356303, 
    max_depth=-1, 
    n_estimators=387, 
    num_leaves=130,
    scale_pos_weight=scale_pos_weight  
)

print(f"Class distribution in training set: {class_counts}")
print(f"Scale_pos_weight: {scale_pos_weight}")

# 모델 학습
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# 평가지표 계산 함수 작성
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
    print(f"{metric}: {value:.4f}")
