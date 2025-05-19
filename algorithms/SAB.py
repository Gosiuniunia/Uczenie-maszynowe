import numpy as np
from sklearn.base import clone

class StrengthenedAdaBoostClassifier:
    def __init__(self, n_estimators, estimator, learning_rate = 1.0):
        self.n_estimators = n_estimators
        self.estimator = estimator
        self.learning_rate = learning_rate
        self.classifiers = []
        self.alphas = []

    def fit(self, X, y):
        self.N = len(X)
        self.k = 0.2
        self.theta = 0.5

        y = np.where(y == 0, -1, 1)
        classes, class_counts = np.unique(y, return_counts=True)
        imbalanced_ratio = class_counts.min() / class_counts.max()
        self.b = 1/imbalanced_ratio
        omega = np.array([1 / self.N for _ in range(self.N)])
        for i in range(self.n_estimators):
            clf = clone(self.estimator)
            clf.fit(X, y, sample_weight=omega)
            y_pred = clf.predict(X)
            alpha_m = self._count_alpha_value(y_pred, y, omega)
            self.classifiers.append(clf)
            self.alphas.append(self.learning_rate*alpha_m)
            omega = self._upadate_weights(omega, alpha_m, y, y_pred)


    def predict(self, X):
        final_score = np.zeros(len(X))
        for alpha, clf in zip(self.alphas, self.classifiers):
            pred = clf.predict(X)
            final_score += alpha * pred
        return np.where(np.sign(final_score) == -1, 0, 1)

    def _count_alpha_value(self, y_pred, y, omega):
        incorrect = y_pred != y
        epsilon_m = np.sum(omega[incorrect])
        epsilon_m = max(epsilon_m, 1e-100)
        positive_mask = y == 1
        Q_m = np.sum(omega[positive_mask])
        P_m = np.sum(omega[positive_mask & (y_pred == y)])
        delta_m = P_m / Q_m if Q_m > 0 else 0.0
        value = 0.5*(1 - (2 * delta_m)/(self.b + 1))
        if epsilon_m < value:
            term1 = 0.5 * np.log((1 - epsilon_m) / epsilon_m)
            term2 = self.k * (1 - np.exp(1 - self.b)) * np.exp(self.theta * (2 * delta_m - 1))
            return term1 + term2
        else:
            term1 = 0.5 * np.log((1 - epsilon_m) / epsilon_m)
            term2 = (np.exp(delta_m - 0.5) + 0.5) / (0.5 - epsilon_m) + 0.5*(self.b +1)/(delta_m - 0.5)
            return term1*term2
        
    def _upadate_weights(self, omega, alpha_m, y, y_pred):
        exponent = -alpha_m * y * y_pred
        exponent = np.clip(exponent, -100, 100)
        new_omega = omega * np.exp(exponent)
        new_omega = np.clip(new_omega, 1e-10, None)
        new_omega /= np.sum(new_omega)
        return new_omega