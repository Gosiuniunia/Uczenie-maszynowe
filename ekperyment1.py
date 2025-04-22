import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from ensemble import StrengthenedAdaBoostClassifier
from xgboost import XGBClassifier

from SWSEL import SWSEL
from ensemble import StrengthenedAdaBoostClassifier
from VAO import VAO


X, y = load_breast_cancer(return_X_y=True)
y = pd.Series(y)
class_counts = y.value_counts()
rskf = RepeatedStratifiedKFold(n_repeats=5, n_splits=2, random_state=100)

def apply_oversampling(method, X_train, y_train, alpha = None):
    if method == "SMOTE":
        sampler = SMOTE(sampling_strategy=0.75, random_state=100)
    elif method == "RUS":
        sampler = RandomOverSampler(sampling_strategy=0.75, random_state=100)
    elif method == "ADASYN":
        sampler = ADASYN(sampling_strategy=0.75, random_state=100)
    elif method == "VAO":
        sampler = VAO(sampling_strategy=0.75, alpha=alpha)
    else:
        return X_train, y_train
    return sampler.fit_resample(X_train, y_train)

def tune_single_param(method_name, classifier_name, M_list, oversampling_type=None):
    results = []
    for M in M_list:
        if classifier_name in [SWSEL, BaggingClassifier]:
            clf = classifier_name(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=M)
        else:
            clf = classifier_name(n_estimators=M)
        for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            X_res, y_res = apply_oversampling(oversampling_type, X_train, y_train)

            clf.fit(X_res, y_res)
            y_pred = clf.predict(X_test)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            g_mean = geometric_mean_score(y_test, y_pred)

            results.append({
                'method': method_name,
                'M': M,
                'lr': None,
                'alpha': None,
                'beta': None,
                'fold': fold_idx,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'G-Mean': g_mean
            })
    return results

def tune_M_lr(method_name, classifier_name, M_list, lr_list, oversampling_type=None):
    results = []
    for M in M_list:
        for lr in lr_list:
            if classifier_name in [AdaBoostClassifier]:
                clf = classifier_name(estimator = DecisionTreeClassifier(max_depth=1), n_estimators = M, learning_rate = lr, algorithm="SAMME")
            else:
                clf = XGBClassifier(n_estimators=M, learning_rate=lr,  max_depth=1)
            for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
                X_train, y_train = X[train_idx], y[train_idx]
                X_test, y_test = X[test_idx], y[test_idx]
                X_res, y_res = apply_oversampling(oversampling_type, X_train, y_train)

                clf.fit(X_res, y_res)
                y_pred = clf.predict(X_test)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                g_mean = geometric_mean_score(y_test, y_pred)

                results.append({
                    'method': method_name,
                    'M': M,
                    'lr': lr,
                    'alpha': None,
                    'beta': None,
                    'fold': fold_idx,
                    'Precision': precision,
                    'Recall': recall,
                    'F1 Score': f1,
                    'G-Mean': g_mean
                })
    return results

def tune_M_lr_alpha(method_name, classifier_name, M_list, lr_list, alpha_list, oversampling_type=None):
    results = []
    for M in M_list:
        for lr in lr_list:
            for alpha in alpha_list:
                beta = 1 - alpha
                clf = classifier_name(estimator = DecisionTreeClassifier(max_depth=1), n_estimators = M, learning_rate = lr)
                for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
                    X_train, y_train = X[train_idx], y[train_idx]
                    X_test, y_test = X[test_idx], y[test_idx]
                    X_res, y_res = apply_oversampling(oversampling_type, X_train, y_train, alpha)

                    clf.fit(X_res, y_res)
                    y_pred = clf.predict(X_test)
                    precision = precision_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    g_mean = geometric_mean_score(y_test, y_pred)

                    results.append({
                        'method': method_name,
                        'M': M,
                        'lr': lr,
                        'alpha': alpha,
                        'beta': beta,
                        'fold': fold_idx,
                        'Precision': precision,
                        'Recall': recall,
                        'F1 Score': f1,
                        'G-Mean': g_mean
                    })
    return results



def run_tuning():
    # M_list = [25, 50, 75, 100]
    M_list = [10]
    lr_list = [1]
    # lr_list = [0.1, 0.5, 1, 10]
    alpha_list=[0.1 * i for i in range(1, 2)]
    # alpha_list=[0.1 * i for i in range(1, 10)]
    # ================================
    # 1.1 SWESEL (Bez samplingu, tylko M)
    # swe_results = tune_single_param("SWSEL", SWSEL, M_list, oversampling_type=None)
    
    # ================================
    # 1.2 SMRF (SMOTE + Random Forest, tylko M)
    smrf_results = tune_single_param("SMRF", RandomForestClassifier, M_list, oversampling_type="SMOTE")
    
    # ================================
    # # 1.3 SMB (SMOTE + Bagging, tylko M)
    smb_results = tune_single_param("SMB", BaggingClassifier, M_list, oversampling_type="SMOTE")
    
    # # ================================
    # # 1.4 SMAB (SMOTE + AdaBoost, M i lr)
    smab_results = tune_M_lr("SMAB", AdaBoostClassifier, M_list, lr_list, oversampling_type="SMOTE")
    
    # # ================================
    # # 1.5 RUSAB (RUS + AdaBoost, M i lr)
    rusab_results = tune_M_lr("RUSAB", AdaBoostClassifier, M_list, lr_list, oversampling_type="RUS")

    # # ================================
    # # 1.6 ADXG (ADASYN + XGBoost, M i lr)
    adxg_results = tune_M_lr("ADXG", XGBClassifier, M_list, lr_list, oversampling_type="ADASYN")
    
    # # ================================
    # # 1.7 VASA (VAO + AdaBoost, M, lr i alpha)
    vasa_results = tune_M_lr_alpha("VASA", StrengthenedAdaBoostClassifier, M_list, lr_list, alpha_list, oversampling_type="VAO")
    
    # ================================
    # Zbieranie wszystkich wyników
    all_results = smrf_results + smb_results + smab_results + rusab_results + adxg_results + vasa_results
    # swe_results
    
    # Przechowywanie wyników w pliku CSV
    df = pd.DataFrame(all_results)
    df.to_csv("gridsearch_results.csv", index=False)
    return df


run_tuning()