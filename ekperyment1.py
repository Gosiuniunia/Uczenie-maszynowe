import numpy as np
import pandas as pd
import itertools
from sklearn.metrics import precision_score, recall_score, f1_score
from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from ensemble import StrengthenedAdaBoostClassifier
from xgboost import XGBClassifier
from preprocessing import preprocess_data

from SWSEL import SWSEL
from ensemble import StrengthenedAdaBoostClassifier
from VAO import VAO


file_path = "PCOS_data_without_infertility.xlsx"
X, y = preprocess_data(file_path)
rskf = RepeatedStratifiedKFold(n_repeats=2, n_splits=5, random_state=100)

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
    n_folds = rskf.get_n_splits()
    n_params = len(M_list)
    
    precisions = np.zeros((n_params, n_folds))
    recalls = np.zeros_like(precisions)
    f1s = np.zeros_like(precisions)
    gmeans = np.zeros_like(precisions)

    for param_idx, M in enumerate(M_list):
        print(param_idx, M)
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

            precisions[param_idx, fold_idx] = precision_score(y_test, y_pred)
            recalls[param_idx, fold_idx] = recall_score(y_test, y_pred)
            f1s[param_idx, fold_idx] = f1_score(y_test, y_pred)
            gmeans[param_idx, fold_idx] = geometric_mean_score(y_test, y_pred)

    np.save(f"{method_name.lower()}_precision.npy", precisions)
    np.save(f"{method_name.lower()}_recall.npy", recalls)
    np.save(f"{method_name.lower()}_f1_score.npy", f1s)
    np.save(f"{method_name.lower()}_g-mean.npy", gmeans)

def tune_M_st(method_name, classifier_name, M_list, step_list, oversampling_type=None):
    n_folds = rskf.get_n_splits()
    n_params = len(M_list) * len(step_list)

    precisions = np.zeros((n_params, n_folds))
    recalls = np.zeros_like(precisions)
    f1s = np.zeros_like(precisions)
    gmeans = np.zeros_like(precisions)

    for param_idx, (M, st) in enumerate(itertools.product(M_list, step_list)):
        clf = classifier_name(estimator = DecisionTreeClassifier(max_depth=1), n_estimators = M, step_size = st)
        
        for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            X_res, y_res = apply_oversampling(oversampling_type, X_train, y_train)

            clf.fit(X_res, y_res)
            y_pred = clf.predict(X_test)

            precisions[param_idx, fold_idx] = precision_score(y_test, y_pred)
            recalls[param_idx, fold_idx] = recall_score(y_test, y_pred)
            f1s[param_idx, fold_idx] = f1_score(y_test, y_pred)
            gmeans[param_idx, fold_idx] = geometric_mean_score(y_test, y_pred)

    np.save(f"{method_name.lower()}_precision.npy", precisions)
    np.save(f"{method_name.lower()}_recall.npy", recalls)
    np.save(f"{method_name.lower()}_f1_score.npy", f1s)
    np.save(f"{method_name.lower()}_g-mean.npy", gmeans)

def tune_M_lr(method_name, classifier_name, M_list, lr_list, oversampling_type=None):
    n_folds = rskf.get_n_splits()
    n_params = len(M_list) * len(lr_list)

    precisions = np.zeros((n_params, n_folds))
    recalls = np.zeros_like(precisions)
    f1s = np.zeros_like(precisions)
    gmeans = np.zeros_like(precisions)

    for param_idx, (M, lr) in enumerate(itertools.product(M_list, lr_list)):
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

            precisions[param_idx, fold_idx] = precision_score(y_test, y_pred)
            recalls[param_idx, fold_idx] = recall_score(y_test, y_pred)
            f1s[param_idx, fold_idx] = f1_score(y_test, y_pred)
            gmeans[param_idx, fold_idx] = geometric_mean_score(y_test, y_pred)

    np.save(f"{method_name.lower()}_precision.npy", precisions)
    np.save(f"{method_name.lower()}_recall.npy", recalls)
    np.save(f"{method_name.lower()}_f1_score.npy", f1s)
    np.save(f"{method_name.lower()}_g-mean.npy", gmeans)

def tune_M_lr_alpha(method_name, classifier_name, M_list, lr_list, alpha_list, oversampling_type=None):
    n_folds = rskf.get_n_splits()
    n_params = len(M_list) * len(lr_list) * len(alpha_list)

    precisions = np.zeros((1, n_params, n_folds))
    recalls = np.zeros_like(precisions)
    f1s = np.zeros_like(precisions)
    gmeans = np.zeros_like(precisions)

    for param_idx, (M, lr, alpha) in enumerate(itertools.product(M_list, lr_list, alpha_list)):
        clf = classifier_name(estimator = DecisionTreeClassifier(max_depth=1), n_estimators = M, learning_rate = lr)
        for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            X_res, y_res = apply_oversampling(oversampling_type, X_train, y_train, alpha)

            clf.fit(X_res, y_res)
            y_pred = clf.predict(X_test)

            precisions[0, param_idx, fold_idx] = precision_score(y_test, y_pred)
            recalls[0, param_idx, fold_idx] = recall_score(y_test, y_pred)
            f1s[0, param_idx, fold_idx] = f1_score(y_test, y_pred)
            gmeans[0, param_idx, fold_idx] = geometric_mean_score(y_test, y_pred)

    np.save(f"{method_name.lower()}_precision.npy", precisions)
    np.save(f"{method_name.lower()}_recall.npy", recalls)
    np.save(f"{method_name.lower()}_f1_score.npy", f1s)
    np.save(f"{method_name.lower()}_g-mean.npy", gmeans)



def run_tuning():
    M_list = [25, 50, 75, 100]
    # M_list = [50]
    # lr_list = [1]
    lr_list = [0.1, 0.5, 1, 10]
    # alpha_list=[0.1 * i for i in range(1, 2)]
    # print(alpha_list)
    alpha_list=[0.1 * i for i in range(1, 10)]
    # step_list = [1]
    step_list = [1, 5, 10, 20]
    # ================================
    # 1.1 SWESEL (Bez samplingu, tylko M)
    tune_single_param("SWSEL", SWSEL, M_list, oversampling_type=None)
    
    # ================================
    # 1.2 SMRF (SMOTE + Random Forest, tylko M)
    tune_single_param("SMRF", RandomForestClassifier, M_list, oversampling_type="SMOTE")
    
    # ================================
    # # 1.3 SMB (SMOTE + Bagging, tylko M)
    tune_single_param("SMB", BaggingClassifier, M_list, oversampling_type="SMOTE")
    
    # # ================================
    # # 1.4 SMAB (SMOTE + AdaBoost, M i lr)
    tune_M_lr("SMAB", AdaBoostClassifier, M_list, lr_list, oversampling_type="SMOTE")
    
    # # ================================
    # # 1.5 RUSAB (RUS + AdaBoost, M i lr)
    tune_M_lr("RUSAB", AdaBoostClassifier, M_list, lr_list, oversampling_type="RUS")

    # # ================================
    # # 1.6 ADXG (ADASYN + XGBoost, M i lr)
    tune_M_lr("ADXG", XGBClassifier, M_list, lr_list, oversampling_type="ADASYN")
    
    # # ================================
    # # 1.7 VASA (VAO + AdaBoost, M, lr i alpha)
    tune_M_lr_alpha("VASA", StrengthenedAdaBoostClassifier, M_list, lr_list, alpha_list, oversampling_type="VAO")


run_tuning()