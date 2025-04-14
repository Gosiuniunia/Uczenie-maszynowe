import numpy as np
from sklearn.base import clone

class StrenththedAdaBoostClassifier:
    def __init__(self, n_estimators, imbalanced_ratio, estimator):
        self.n_estimators = n_estimators
        self.imbalanced_ratio = imbalanced_ratio
        self.estimator = estimator
        self.classifiers = []
        self.alphas = []

    def fit(self, X, y):
        self.N = len(X)
        y = np.where(y == 0, -1, 1)

        omega = [1 / self.N for n in range(self.N)]
        classiefiers = [self.estimator for _ in range(self.N)]
        for i in range(self.n_estimators):
            clf = clone(self.estimator)
            clf.fit(X, y, sample_weight=omega)
            y_pred = clf.predict(X)
            alpha_m = self._count_alpha_value(y_pred, y, omega)
            self.classifiers.append(clf)
            self.alphas.append(alpha_m)
            omega = self._upadate_weights(omega, alpha_m, y, y_pred)


    def predict(self, X):
        final_score = np.zeros(len(X))
        for alpha, clf in zip(self.alphas, self.classifiers):
            pred = clf.predict(X)
            final_score += alpha * pred
        return np.where(np.sign(final_score) == -1, 0, 1)

    def _count_alpha_value(self, y_pred, y, omega):
        epsilon_m = sum(omega[j] for j in range(self.N) if y_pred[j] != y[j])
        Q_m = sum(omega[j] for j in range(self.N) if y[j] == 1)
        P_m = sum(omega[j] for j in range(self.N) if y[j] == 1 and y_pred[j] == y[j])
        delta_m = P_m / Q_m
        value = 0.5*(1 - (2 * delta_m) / (self.imbalanced_ratio + 1))
        if epsilon_m < value:
            term1 = 0.5 * np.log((1 - epsilon_m) / epsilon_m)
            term2 = self.k * (1 - np.exp(1 - self.imbalanced_ratio)) * np.exp(self.theta * (2 * delta_m - 1))
            return term1 + term2
        else:
            term1 = 0.5 * np.log((1 - epsilon_m) / epsilon_m)
            term2 = (np.exp(delta_m - 0.5) + 0.5) / (0.5 - epsilon_m)
            return term1*term2
        
    def _upadate_weights(self, omega, alpha_m, y, y_pred):
        new_omega = omega * np.exp(-alpha_m * y * y_pred)
        new_omega /= np.sum(new_omega)
        return new_omega

from sklearn.datasets import make_classification
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE

X, y = make_classification(
    n_samples=100,
    n_features=5,
    n_classes=2,
    n_clusters_per_class=1,
    weights=[0.7, 0.3],
    random_state=42,
)

x = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
y = pd.Series(y)

# Oversampling (tu SMOTE jako VAO)
VAO = SMOTE(random_state=42)
X_resampled, y_resampled = VAO.fit_resample(x, y)

print(y_resampled.value_counts())

# Użycie klasyfikatora
clf = StrenththedAdaBoostClassifier(
    n_estimators=3, imbalanced_ratio=1, estimator=DecisionTreeClassifier(max_depth=1)
)

clf.fit(X_resampled, y_resampled)

predictions = clf.predict(X_resampled)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_resampled, predictions)
print(f'Accuracy: {accuracy:.2f}')