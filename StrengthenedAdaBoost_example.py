from sklearn.datasets import make_classification
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from VAO import VAO
from ensemble import StrengthenedAdaBoostClassifier
from sklearn.metrics import accuracy_score

X, y = make_classification(
    n_samples=1000,
    n_features=5,
    n_classes=2,
    weights=[0.7, 0.3],
    random_state=4262,
)

x = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
y = pd.Series(y)

# Oversampling (tu SMOTE jako VAO)
vao = VAO()
X_resampled, y_resampled = vao.fit_resample(x, y)
# Użycie klasyfikatora
clf = StrengthenedAdaBoostClassifier(
    n_estimators=100, estimator=DecisionTreeClassifier(max_depth=2)
)

clf.fit(X_resampled, y_resampled)

predictions = clf.predict(X_resampled)


accuracy = accuracy_score(y_resampled, predictions)
print(f'Accuracy: {accuracy:.2f}')