import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from sklearn.datasets import make_blobs
from xie_beni_index import xie_beni_index
import pandas as pd


X, _ = make_blobs(n_samples=300, centers=4, n_features=2, random_state=42)
X = pd.DataFrame(X)
X_T = X.T

c_values = list(range(2, 11))
xb_values = []

for c in c_values:
    cntr, U, _, _, _, _, _ = fuzz.cluster.cmeans(X_T, c, m=2, error=0.005, maxiter=1000, init=None)
    xb = xie_beni_index(X, U, cntr)
    xb_values.append(xb)

plt.figure(figsize=(8, 5))
plt.plot(c_values, xb_values, marker='o', color = "hotpink")
plt.title('Wartość indeksu Xie-Beni od liczby klastrów')
plt.xlabel('Liczba klastrów (C)')
plt.ylabel('Xie-Beni Index')
plt.grid(True)
plt.show()
