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
    def __init__(self, sampling_strategy=0.8, k=5, alpha=0.5):
        self.sampling_strategy = sampling_strategy
        self.k = k
        self.alpha = alpha
        self.beta = 1 - alpha

    def fit_resample(self, x, y):
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        if not isinstance(x, pd.DataFrame):
            x = pd.DataFrame(x)
        y = y.reset_index(drop=True)
        x = x.reset_index(drop=True)
        class_counts = y.value_counts()
        majority_class = class_counts.idxmax()
        minority_class = class_counts.idxmin()
        n_maj = class_counts[majority_class]
        n_min = class_counts[minority_class]
        G = floor(n_maj * self.sampling_strategy - n_min)
        x_resampled, y_resampled = self.clean_samples(x, y, minority_class)
        x, y = self.clean_samples(x_resampled, y_resampled, majority_class)
        class_counts = y.value_counts()
        n_maj = class_counts[majority_class]
        n_min = class_counts[minority_class]
        minority_samples = x[y == minority_class]
        self.num_clusters = self.count_cluster_number(minority_samples)

        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
        kmeans.fit(minority_samples)

        cluster_centers = kmeans.cluster_centers_
        samples_per_cluster = [minority_samples[kmeans.labels_ == i] for i in range(self.num_clusters)]
        majority_samples = x[y == majority_class]

        L_hat = self.count_L_hat(samples_per_cluster, majority_samples)
        S_hat = self.count_S_hat(samples_per_cluster, cluster_centers)
        g = self.count_g(G, L_hat, S_hat)
        new_x = self.generate_samples(samples_per_cluster, cluster_centers, g)
        new_x_df = pd.DataFrame(new_x, columns=x.columns)
        X = pd.concat([x, new_x_df], ignore_index=True)
        new_y = pd.Series([minority_class] * len(new_x))
        y = pd.concat([y, new_y], ignore_index=True)
        return X, y

    def clean_samples(self, x, y, label_to_clean):
        nneigh = NearestNeighbors(n_neighbors=self.k)
        nneigh.fit(x)
        maj_class = x[y == (1 - label_to_clean)]
        indices_maj_class = maj_class.index
        min_class = x[y == label_to_clean]
        indices_to_remove = []
        for idx, x_i in min_class.iterrows():
            distances, indices = nneigh.kneighbors(x_i.to_frame().T)
            indices_set = set(indices.flatten())
            maj_class_set = set(indices_maj_class)
            if indices_set.issubset(maj_class_set):
                indices_to_remove.append(idx)

        x_resampled = x.drop(indices_to_remove)
        y_resampled = y.drop(indices_to_remove)
        return (x_resampled, y_resampled)

    def count_cluster_number(self, X, c_min = 2, c_max = 11):
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

    def count_L_hat(self, samples_per_cluster, majority_samples):
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

    def count_S_hat(self, samples_per_cluster, cluster_centers):
        S_list = []
        for i in range(self.num_clusters):
            cluster_samples = samples_per_cluster[i]
            A_i = cluster_centers[i]
            Si_list = []
            for index, sample in cluster_samples.iterrows():
                dot_product = np.dot(sample, A_i)
                norm_sample = np.linalg.norm(sample)
                norm_Ai = np.linalg.norm(A_i)
                S_i = dot_product / (norm_sample * norm_Ai)
                Si_list.append(S_i)
            Si_prim = np.mean(Si_list)
            S_list.append(1 / Si_prim)

        Si_max = max(S_list)
        Si_min = min(S_list)
        S_hat = []
        for Si_i in S_list:
            Si_hat = (Si_i - Si_min) / (Si_max - Si_min)
            S_hat.append(Si_hat)
        return S_hat

    def count_g(self, G, L_hat, S_hat):
        W = [0 for i in range(self.num_clusters)]
        for i in range(self.num_clusters):
            W[i] = self.alpha * L_hat[i] + self.beta * S_hat[i]

        W_sum = sum(W)
        W_hat = [wi / W_sum for wi in W]
        g = [G * wi_hat for wi_hat in W_hat]
        return g

    def generate_samples(self, samples_per_cluster, cluster_centers, g):
        new_samples = []
        for i in range(self.num_clusters):
            cluster_samples = samples_per_cluster[i]
            A_i = cluster_centers[i]
            for _ in range(floor(g[i])):
                x_ij = cluster_samples.sample(n=1).values.flatten()

                # Losowy wybór dwóch sąsiednich próbek
                x_il = cluster_samples.sample(n=1).values.flatten()
                x_is = cluster_samples.sample(n=1).values.flatten()

                t_k_ip1 = x_ij + rand() * (x_il - x_ij)
                t_k_ip2 = x_ij + rand() * (x_is - x_ij)

                t_k_ip = t_k_ip1 + np.random.rand() * (t_k_ip2 - t_k_ip1)
                y_k_ip = A_i + np.random.rand() * (t_k_ip - A_i)
                new_samples.append(y_k_ip)
        return new_samples
