from sklearn.datasets import make_classification
from VAO import VAO
import pandas as pd



X, y = make_classification(n_samples=100, n_features=5, n_classes=2, 
                            n_clusters_per_class=1, weights=[0.7, 0.3], random_state=42)

x = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
y = pd.Series(y)

a = VAO()
X, y = a.fit_resample(x, y)

print(y.value_counts())