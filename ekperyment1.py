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


# file_path = "PCOS_data_without_infertility.xlsx"
# X, y = preprocess_data(file_path)
# rskf = RepeatedStratifiedKFold(n_repeats=5, n_splits=5, random_state=100)

def apply_oversampling(method, X_train, y_train, alpha = None):
    if method == "SMOTE":
        sampler = SMOTE(sampling_strategy=0.75, random_state=100)
    elif method == "RUS":
        sampler = RandomOverSampler(sampling_strategy=0.75, random_state=100)
    elif method == "VAO":
        sampler = VAO(sampling_strategy=0.75, alpha=alpha, random_state=100)
    else:
        return X_train, y_train
    return sampler.fit_resample(X_train, y_train)

def tune_M(method_name, classifier_name, M_list, rskf, X, y, oversampling_type=None):
    n_folds = rskf.get_n_splits()
    n_params = len(M_list)
    
    precisions = np.zeros((n_params, n_folds))
    recalls = np.zeros_like(precisions)
    f1s = np.zeros_like(precisions)
    gmeans = np.zeros_like(precisions)

    for param_idx, M in enumerate(M_list):
        if classifier_name == SWSEL:
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

def tune_M_lr(method_name, classifier_name, M_list, lr_list, rskf, X, y, oversampling_type=None):
    n_folds = rskf.get_n_splits()
    n_params = len(M_list) * len(lr_list)

    precisions = np.zeros((n_params, n_folds))
    recalls = np.zeros_like(precisions)
    f1s = np.zeros_like(precisions)
    gmeans = np.zeros_like(precisions)

    for param_idx, (M, lr) in enumerate(itertools.product(M_list, lr_list)):
        if classifier_name == AdaBoostClassifier:
            clf = classifier_name(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=M, learning_rate=lr, algorithm="SAMME")
        else:
            clf = classifier_name(estimator = DecisionTreeClassifier(max_depth=1), n_estimators = M, learning_rate = lr)
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

def tune_M_alpha(method_name, classifier_name, M_list, alpha_list, rskf, X, y, oversampling_type=None):
    n_folds = rskf.get_n_splits()
    n_params = len(M_list) * len(alpha_list)

    precisions = np.zeros((n_params, n_folds))
    recalls = np.zeros_like(precisions)
    f1s = np.zeros_like(precisions)
    gmeans = np.zeros_like(precisions)

    for param_idx, (M, alpha) in enumerate(itertools.product(M_list, alpha_list)):
        if classifier_name == SWSEL:
            clf = classifier_name(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=M)
        else:
            clf = classifier_name(n_estimators=M)
        for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            X_res, y_res = apply_oversampling(oversampling_type, X_train, y_train, alpha)

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


def tune_M_lr_alpha(method_name, classifier_name, M_list, lr_list, alpha_list, rskf, X, y, oversampling_type=None):
    n_folds = rskf.get_n_splits()
    n_params = len(M_list) * len(lr_list) * len(alpha_list)

    precisions = np.zeros((n_params, n_folds))
    recalls = np.zeros_like(precisions)
    f1s = np.zeros_like(precisions)
    gmeans = np.zeros_like(precisions)

    for param_idx, (M, lr, alpha) in enumerate(itertools.product(M_list, lr_list, alpha_list)):
        if classifier_name == AdaBoostClassifier:
            clf = classifier_name(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=M, learning_rate=lr, algorithm="SAMME")
        else:
            clf = classifier_name(estimator = DecisionTreeClassifier(max_depth=1), n_estimators = M, learning_rate = lr)
        for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            X_res, y_res = apply_oversampling(oversampling_type, X_train, y_train, alpha)

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


def run_tuning():
    file_path = "PCOS_data_without_infertility.xlsx"
    X, y = preprocess_data(file_path)
    rskf = RepeatedStratifiedKFold(n_repeats=5, n_splits=5, random_state=100)
    M_list = [25, 50, 75, 100]
    lr_list = [0.1, 0.5, 1, 10]
    alpha_list = [round(0.1 * i,1) for i in range(1, 11)]

    classifiers = {
        # "SWSEL": SWSEL,
        # "RF": RandomForestClassifier,
        "AB": AdaBoostClassifier,
        "SAB": StrengthenedAdaBoostClassifier
    }

    oversamplings = ["NONE", "SMOTE", "RUS", "VAO"]

    for clf_name, clf_class in classifiers.items():
        for sampling in oversamplings:
            method_name = f"{clf_name}_{sampling}"
            if sampling == "VAO":
                if clf_class in [AdaBoostClassifier, StrengthenedAdaBoostClassifier]:
                    tune_M_lr_alpha(method_name, clf_class, M_list, lr_list, alpha_list, rskf, X, y, sampling)
                else:
                    tune_M_alpha(method_name, clf_class, M_list, alpha_list, rskf, X, y, sampling)

            else:
                if clf_class in [AdaBoostClassifier, StrengthenedAdaBoostClassifier]:
                    tune_M_lr(method_name, clf_class, M_list, lr_list, rskf, X, y, sampling)
                else:
                    tune_M(method_name, clf_class, M_list, rskf, X, y, sampling)
run_tuning()