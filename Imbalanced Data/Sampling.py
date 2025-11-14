from collections import Counter
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import pandas as pd
# 데이터 불러오기
df = pd.read_csv("1주_실습데이터.csv")

# 불필요한 열 제거
df = df.drop(columns=['X2','X4', 'X13', 'X18', 'X19', 'X20'])
# Y의 클래스 분포 확인
print("클래스 분포:", Counter(df['Y']))

# 독립 변수(X)와 종속 변수(Y) 분리
X = df.drop('Y', axis=1)
Y = df['Y']

# SMOTE 적용
smote = SMOTE(random_state=42)
X_res, Y_res = smote.fit_resample(X, Y)

# SMOTE 적용 후 클래스 분포 확인
print("SMOTE 적용 후 클래스 분포:", Counter(Y_res))

# 클래스 분포 시각화
plt.figure(figsize=(6,4))
plt.bar(Counter(Y_res).keys(), Counter(Y_res).values())
plt.title('클래스 분포 (SMOTE 적용 후)')
plt.xlabel('클래스')
plt.ylabel('샘플 수')
plt.show()

# Borderline-SMOTE 적용
from imblearn.over_sampling import BorderlineSMOTE
import matplotlib.pyplot as plt
from collections import Counter

borderline_smote = BorderlineSMOTE(random_state=42)
X_border_smote, Y_border_smote = borderline_smote.fit_resample(X, Y)

# Borderline-SMOTE 적용 후 클래스 분포 확인
print("Borderline-SMOTE 적용 후 클래스 분포:", Counter(Y_border_smote))

# 클래스 분포 시각화
plt.figure(figsize=(6,4))
plt.bar(Counter(Y_border_smote).keys(), Counter(Y_border_smote).values())
plt.title('클래스 분포 (Borderline-SMOTE 적용 후)')
plt.xlabel('클래스')
plt.ylabel('샘플 수')
plt.show()

# ADASYN 적용
from imblearn.over_sampling import ADASYN

adasyn = ADASYN(random_state=42)
X_adasyn, Y_adasyn = adasyn.fit_resample(X, Y)

# ADASYN 적용 후 클래스 분포 확인
print("ADASYN 적용 후 클래스 분포:", Counter(Y_adasyn))

# 클래스 분포 시각화
plt.figure(figsize=(6,4))
plt.bar(Counter(Y_adasyn).keys(), Counter(Y_adasyn).values())
plt.title('클래스 분포 (ADASYN 적용 후)')
plt.xlabel('클래스')
plt.ylabel('샘플 수')
plt.show()

# CURE_SMOTE