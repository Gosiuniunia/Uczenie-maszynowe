import numpy as np

def xie_beni_index(X, U, centroids, m=2):
    n = X.shape[0]
    C = centroids.shape[0]
    numerator = np.sum((U ** m) * np.linalg.norm(X.to_numpy()[:, None] - centroids, axis=2).T ** 2)
    min_dist = np.inf
    for i in range(C):
        for j in range(C):
            if i != j:
                dist = np.linalg.norm(centroids[i] - centroids[j]) ** 2
                if dist < min_dist:
                    min_dist = dist
    
    xb_index = numerator / (n * min_dist)
    return xb_index