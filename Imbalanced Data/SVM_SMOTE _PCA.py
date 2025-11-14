import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.decomposition import PCA

# 데이터 불러오기
df = pd.read_csv("1주_실습데이터.csv")

# 불필요한 열 제거
df = df.drop(columns=['X4', 'X13', 'X18', 'X19', 'X20'])

# 독립변수(X)와 종속변수(y) 분리
X = df.drop("Y", axis=1)
y = df['Y']

# train/test 셋 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# StandardScaler 적용
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA로 차원 축소 
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# PCA 적용 후 스케일된 데이터를 DataFrame으로 변환
X_train_pca = pd.DataFrame(X_train_pca)
X_test_pca = pd.DataFrame(X_test_pca)

# 차원 축소 후 SMOTE 적용 전 클래스 분포 확인
print(f"Original dataset shape: {Counter(y_train)}")

# SMOTE 적용
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_pca, y_train)

#-------------------------------------------------------------------
# 평가지표 계산 함수 작성
import time
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, accuracy_score
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import roc_curve, auc

#!pip install imblearn

def Eval(y_true, y_pred, y_pred_proba=None):
    # 학습 시작 시간 기록
    start_time = time.time()
    
    # F1 score 계산
    f1 = f1_score(np.round(y_true), np.round(y_pred), average='weighted')
    
    # Geometric Mean (G-Mean) 계산
    gmean = geometric_mean_score(np.round(y_true), np.round(y_pred), average='weighted')
    
    # Global Accuracy 계산
    accuracy = accuracy_score(y_true, y_pred)
    
    # AUC 계산 
    auc_score = roc_auc_score(y_true, y_pred_proba[:, 1]) if y_pred_proba is not None else None
    
    # 혼동행렬 계산
    conf_matrix = confusion_matrix(y_true, y_pred)

    
    return {
        "F1 Score": f1,
        "G-Mean": gmean,
        "Global Accuracy": accuracy,
        "AUC": auc_score,
        "Confusion Matrix": conf_matrix
    }
#---------------------------------------------------------------------
# SVM
from sklearn.svm import SVC

# SVM 모델 생성
svm_model = SVC(kernel='rbf', random_state=42, probability=True)

# SVM 모델 훈련
svm_model.fit(X_train_pca, y_train)

# 검증 데이터로 예측
y_pred = svm_model.predict(X_test_pca)
y_pred_proba = svm_model.predict_proba(X_test_pca)

# 성능 평가
results = Eval(y_test, y_pred, y_pred_proba)

# 결과 출력
for metric, value in results.items():
    print(f"{metric}: {value}")

