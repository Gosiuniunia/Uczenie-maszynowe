import pandas as pd
from math import floor
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import numpy as np
from numpy.random import rand
import skfuzzy as fuzz
from xie_beni_index import xie_beni_index

np.random.seed(100)

class VAO:
    def __init__(self, sampling_strategy=0.8, k=5, alpha=0.5, random_state = 100):
        self.sampling_strategy = sampling_strategy
        self.k = k
        self.alpha = alpha
        self.beta = 1 - alpha
        self.random_state = random_state
        self.rng = np.random.default_rng(self.random_state)


    def fit_resample(self, X, y):
        classes, counts = np.unique(y, return_counts=True)
        majority_class = classes[np.argmax(counts)]
        minority_class = classes[np.argmin(counts)]

        n_maj = counts[np.argmax(counts)]
        n_min = counts[np.argmin(counts)]
        G = floor(n_maj * self.sampling_strategy - n_min)

        X, y = self._clean_samples(X, y, minority_class)
        X, y = self._clean_samples(X, y, majority_class)

        minority_samples = X[y == minority_class]
        majority_samples = X[y == majority_class]

        self.num_clusters = self._count_cluster_number(minority_samples)

        kmeans = KMeans(n_clusters=self.num_clusters, random_state=self.random_state)
        kmeans.fit(minority_samples)
        cluster_centers = kmeans.cluster_centers_

        samples_per_cluster = [minority_samples[kmeans.labels_ == i] for i in range(self.num_clusters)]

        L_hat = self._count_L_hat(samples_per_cluster, majority_samples)
        S_hat = self._count_S_hat(samples_per_cluster, cluster_centers)
        g = self._count_g(G, L_hat, S_hat)

        new_x = self._generate_samples(samples_per_cluster, cluster_centers, g)
        X_res = np.vstack([X, new_x])
        y_res = np.concatenate([y, np.full(len(new_x), minority_class)])
        return X_res, y_res

    def _clean_samples(self, x, y, label_to_clean):
        nneigh = NearestNeighbors(n_neighbors=self.k)
        nneigh.fit(x)
        
        maj_class_indices = np.where(y != label_to_clean)[0]
        min_class_indices = np.where(y == label_to_clean)[0]

        indices_to_remove = []
        for idx in min_class_indices:
            x_i = x[idx].reshape(1, -1)
            _, indices = nneigh.kneighbors(x_i)
            if set(indices.flatten()).issubset(set(maj_class_indices)):
                indices_to_remove.append(idx)

        keep_mask = np.ones(len(y), dtype=bool)
        keep_mask[indices_to_remove] = False
        return x[keep_mask], y[keep_mask]


    def _count_cluster_number(self, X, c_min = 2, c_max = 11):
        c_values = list(range(c_min, c_max))
        xb_values = []
        X_T = X.T

        for c in c_values:
            cntr, U, _, _, _, _, _ = fuzz.cluster.cmeans(
                X_T, c, m=2, error=0.005, maxiter=1000, init=None
            )
            xb = xie_beni_index(X, U, cntr)
            xb_values.append(xb)
        best_c = c_values[np.argmin(xb_values)]

        return best_c

    def _count_L_hat(self, samples_per_cluster, majority_samples):
        xmaj = np.mean(majority_samples, axis=0)
        L_list = []
        for n_cluster in range(self.num_clusters):
            cluster_samples = samples_per_cluster[n_cluster]
            Li = np.linalg.norm(cluster_samples - xmaj, axis=1)
            Li_prim = np.mean(Li)
            L_list.append(1 / Li_prim)
        Li_max = max(L_list)
        Li_min = min(L_list)
        L_hat = []
        for Li_i in L_list:
            Li_hat = (Li_i - Li_min) / (Li_max - Li_min)
            L_hat.append(Li_hat)
        return L_hat

    def _count_S_hat(self, samples_per_cluster, cluster_centers):
        S_list = []
        for i in range(self.num_clusters):
            cluster_samples = samples_per_cluster[i]
            A_i = cluster_centers[i]
            dot_products = np.dot(cluster_samples, A_i)
            norms_samples = np.linalg.norm(cluster_samples, axis=1)
            norm_Ai = np.linalg.norm(A_i)
            similarities = dot_products / (norms_samples * norm_Ai + 1e-10)

            Si_prim = np.mean(similarities)
            S_list.append(1 / (Si_prim + 1e-10))

        Si_max = max(S_list)
        Si_min = min(S_list)
        S_hat = []
        for Si_i in S_list:
            Si_hat = (Si_i - Si_min) / (Si_max - Si_min)
            S_hat.append(Si_hat)
        return S_hat

    def _count_g(self, G, L_hat, S_hat):
        W = [0 for i in range(self.num_clusters)]
        for i in range(self.num_clusters):
            W[i] = self.alpha * L_hat[i] + self.beta * S_hat[i]

        W_sum = np.sum(W)
        W_hat = [wi / W_sum for wi in W]
        g = [G * wi_hat for wi_hat in W_hat]
        return g

    def _generate_samples(self, samples_per_cluster, cluster_centers, g):
        new_samples = []
        for i in range(self.num_clusters):
            cluster_samples = samples_per_cluster[i]
            A_i = cluster_centers[i]
            for _ in range(floor(g[i])):
                idx_ij, idx_il, idx_is = np.random.choice(len(cluster_samples), 3)

                x_ij = cluster_samples[idx_ij]
                x_il = cluster_samples[idx_il]
                x_is = cluster_samples[idx_is]

                t_k_ip1 = x_ij + self.rng.random() * (x_il - x_ij)
                t_k_ip2 = x_ij + self.rng.random() * (x_is - x_ij)

                t_k_ip = t_k_ip1 + self.rng.random() * (t_k_ip2 - t_k_ip1)
                y_k_ip = A_i + self.rng.random() * (t_k_ip - A_i)
                
                new_samples.append(y_k_ip)
        return new_samples
