import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import ADASYN
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from optuna.exceptions import TrialPruned
from sklearn.preprocessing import StandardScaler
import optuna

# 각 모델의 Optuna 최적화 함수 정의
def optimize_rf(trial):
    try:
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            'max_depth': trial.suggest_int('max_depth', 2, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced', 'balanced_subsample'])
        }
        model = RandomForestClassifier(random_state=42, **param)
        model.fit(X_train_res, Y_train_res)
        y_pred = model.predict(X_test)
        return f1_score(Y_test, y_pred)
    except Exception as e:
        print(f"Error in trial for RandomForest: {str(e)}")
        raise TrialPruned()

def optimize_xgb(trial):
    try:
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            'max_depth': trial.suggest_int('max_depth', 2, 20),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.5),
            'subsample': trial.suggest_uniform('subsample', 0.4, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.4, 1.0),
            'colsample_bylevel': trial.suggest_uniform('colsample_bylevel', 0.4, 1.0),
            'colsample_bynode': trial.suggest_uniform('colsample_bynode', 0.4, 1.0),
            'gamma': trial.suggest_loguniform('gamma', 0.001, 10),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 0.01, 10),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 0.01, 10),
            'scale_pos_weight': trial.suggest_loguniform('scale_pos_weight', 0.1, 10)
        }
        model = XGBClassifier(random_state=42, **param)
        model.fit(X_train_res, Y_train_res)
        y_pred = model.predict(X_test)
        return f1_score(Y_test, y_pred)
    except Exception as e:
        print(f"Error in trial for XGBoost: {str(e)}")
        raise TrialPruned()

def optimize_lgbm(trial):
    try:
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            'max_depth': trial.suggest_int('max_depth', -1, 20),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.5),
            'num_leaves': trial.suggest_int('num_leaves', 20, 200),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 200),
            'subsample': trial.suggest_uniform('subsample', 0.4, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.4, 1.0),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 0.01, 10),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 0.01, 10),
            'min_split_gain': trial.suggest_loguniform('min_split_gain', 0.001, 1),
            'min_child_weight': trial.suggest_loguniform('min_child_weight', 0.001, 10),
            'scale_pos_weight': trial.suggest_loguniform('scale_pos_weight', 0.1, 10)
        }
        model = LGBMClassifier(is_unbalance=True, random_state=42, **param)
        model.fit(X_train_res, Y_train_res)
        y_pred = model.predict(X_test)
        return f1_score(Y_test, y_pred)
    except Exception as e:
        print(f"Error in trial for LGBM: {str(e)}")
        raise TrialPruned()

def optimize_catboost(trial):
    try:
        param = {
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.5),
            'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1, 10),
            'border_count': trial.suggest_int('border_count', 32, 255)
        }
        model = CatBoostClassifier(**param, random_seed=42, verbose=0)
        model.fit(X_train_res, Y_train_res)
        y_pred = model.predict(X_test)
        return f1_score(Y_test, y_pred)
    except Exception as e:
        print(f"Error in trial for CatBoost: {str(e)}")
        raise TrialPruned()

# 각 모델에 대한 Optuna 최적화 실행
studies = {}

for model_name, optimization_function in zip(
    ['RandomForest', 'XGBoost', 'LGBM', 'CatBoost'],
    [optimize_rf, optimize_xgb, optimize_lgbm, optimize_catboost]
):
    print(f"Optimizing {model_name}...")
    study = optuna.create_study(direction='maximize')
    study.optimize(optimization_function, n_trials=40)
    studies[model_name] = study
    print(f"Best {model_name} parameters:", study.best_params)
    print(f"Best {model_name} F1 score:", study.best_value)
