import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 데이터 불러오기
df = pd.read_csv("1주_실습데이터.csv")

# 불필요한 열 제거
df = df.drop(columns=['X4', 'X13', 'X18', 'X19', 'X20'])

# 독립변수(X)와 종속변수(y) 분리
X = df.drop("Y", axis=1)
y = df['Y']

# train/test 셋 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# # 칼럼별 왜도 계산
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
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 스케일된 데이터를 DataFrame으로 변환
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)



#-------------------------------------------------------------------
# 평가지표 계산 함수 작성
from sklearn.metrics import f1_score
from imblearn.metrics import geometric_mean_score  # G-Mean
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#!pip install imblearn

def Eval(y_true, y_pred):
    
    # F1 score 계산
    f1 = f1_score(np.round(y_true), np.round(y_pred), average='weighted')
    
    # Geometric Mean (G-Mean) 계산
    gmean = geometric_mean_score(np.round(y_true), np.round(y_pred), average='weighted')
    
    return f1, gmean
#---------------------------------------------------------------------
# SVM
from sklearn.svm import SVC

# SVM 모델 생성
svm_model = SVC(kernel='rbf', random_state=42)

# SVM 모델 훈련
svm_model.fit(X_train, y_train)

# 검증 데이터로 예측
y_pred = svm_model.predict(X_test)

# 성능
f1, gmean = Eval(y_test, y_pred)
print("F1 score:", f1)
print("G-Mean:", gmean)

#---------------------------------------------------------------------
# 시각화
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 혼동 행렬 계산
cm = confusion_matrix(y_test, y_pred)

# 시각화
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
            xticklabels=['Pred Negative', 'Pred Positive'], 
            yticklabels=['True Negative', 'True Positive'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

#-----------------------------------------------------------------------
#왜도 계산 굳이 해야하나,,? 한거랑 안 한거랑 차이 거의 X