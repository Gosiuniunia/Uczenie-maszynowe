import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score

from preprocessing import preprocess_data

class SWSEL:
    def __init__(self, base_classifier, step_size, n_classifiers):

        self.base_classifier = base_classifier
        self.step_size = step_size
        self.n_classifiers = n_classifiers
        self.x_datasets = []
        self.classifiers = []

    def generate_majority_pseudo_sequence(self, X_maj):
        
        majority_center = np.mean(X_maj, axis=0)
        distances = np.sqrt(np.sum(((X_maj - majority_center) / np.std(X_maj, axis=0)) ** 2, axis=1))
        sorted_indices = np.argsort(distances)
        return X_maj[sorted_indices]

    def find_initial_window_position(self, X_maj, X_min):

        minority_center = np.mean(X_min, axis=0)
        distances = np.sqrt(np.sum((X_maj - minority_center) ** 2, axis=1))
        return np.argmin(distances)

    def generate_datasets(self, X_maj, X_min):

        start_idx = self.find_initial_window_position(X_maj, X_min)
        start_idx_temp = start_idx
        majority_pseudo_sequence = self.generate_majority_pseudo_sequence(X_maj)
        window_size = len(X_min)

        if window_size % 2 == 0:
            n_samples = window_size / 2

            if (start_idx + 1 > len(majority_pseudo_sequence) / 2):
                if (len(majority_pseudo_sequence) - start_idx - 1 < n_samples):
                    while (start_idx_temp - n_samples + 1 >= 0):
                        dataset = []
                        if (len(majority_pseudo_sequence) - start_idx_temp - 1 <= n_samples):
                            a_idx = (start_idx_temp - n_samples).astype(int)
                            dataset.append(X_maj[a_idx:])
                            dataset.append(X_min)
                            start_idx_temp -= self.step_size
                            self.x_datasets.append(dataset)
                        
                        else:
                            if (start_idx_temp - n_samples + 1 >= 0):
                                a_idx = (start_idx_temp - n_samples).astype(int)
                                b_idx = (start_idx_temp + n_samples).astype(int)
                                dataset.append(X_maj[a_idx:b_idx])
                                dataset.append(X_min)
                                start_idx_temp -= self.step_size
                                self.x_datasets.append(dataset)
                
                else:
                    cal_dis = len(majority_pseudo_sequence) - start_idx_temp - n_samples
                    while (cal_dis >= 0):
                        dataset = []
                        a_idx = (start_idx_temp - n_samples).astype(int)
                        b_idx = (start_idx_temp + n_samples).astype(int)
                        dataset.append(X_maj[a_idx:b_idx])
                        dataset.append(X_min)
                        start_idx_temp += self.step_size
                        cal_dis -= self.step_size
                        self.x_datasets.append(dataset)
                    start_idx_temp = start_idx
                    while (start_idx_temp - n_samples + 1 >= 0):
                        dataset = []
                        a_idx = (start_idx_temp - n_samples).astype(int)
                        b_idx = (start_idx_temp + n_samples + 1).astype(int)
                        dataset.append(X_maj[a_idx:b_idx])
                        dataset.append(X_min)
                        start_idx_temp -= self.step_size
                        self.x_datasets.append(dataset)

            else:
                if (start_idx + 1 < n_samples):
                    while (start_idx_temp + 1 + n_samples <= len(majority_pseudo_sequence)):
                        dataset = []
                        if (start_idx_temp + 1 <= n_samples):
                            a_idx = (start_idx_temp + n_samples + 1).astype(int)
                            dataset.append(X_maj[:a_idx])
                            dataset.append(X_min)
                            start_idx_temp += self.step_size
                            self.x_datasets.append(dataset)
                        
                        else:
                            if (start_idx_temp + 1 <= len(majority_pseudo_sequence)):
                                a_idx = (start_idx_temp - n_samples).astype(int)
                                b_idx = (start_idx_temp + n_samples + 1).astype(int)
                                dataset.append(X_maj[a_idx:b_idx])
                                dataset.append(X_min)
                                start_idx_temp += self.step_size
                                self.x_datasets.append(dataset)
                
                else:
                    cal_dis = start_idx_temp + 1 - n_samples
                    while (cal_dis >= 0):
                        dataset = []
                        a_idx = (start_idx_temp - n_samples).astype(int)
                        b_idx = (start_idx_temp + n_samples + 1).astype(int)
                        dataset.append(X_maj[a_idx:b_idx])
                        dataset.append(X_min)
                        start_idx_temp -= self.step_size
                        cal_dis -= self.step_size
                        self.x_datasets.append(dataset)
                    start_idx_temp = start_idx
                    while (start_idx_temp + n_samples + 1 <= len(majority_pseudo_sequence)):
                        dataset = []
                        a_idx = (start_idx_temp - n_samples).astype(int)
                        b_idx = (start_idx_temp + n_samples + 1).astype(int)
                        dataset.append(X_maj[a_idx:b_idx])
                        dataset.append(X_min)
                        start_idx_temp += self.step_size
                        self.x_datasets.append(dataset)
        else:
            n_samples = (window_size - 1) / 2

            if (start_idx + 1 > len(majority_pseudo_sequence) / 2):
                if (len(majority_pseudo_sequence) - start_idx - 1 < n_samples):
                    while (start_idx_temp - n_samples + 1 >= 0):
                        dataset = []
                        if (len(majority_pseudo_sequence) - start_idx_temp - 1 <= n_samples):
                            a_idx = (start_idx_temp - n_samples).astype(int)
                            dataset.append(X_maj[a_idx:]) 
                            dataset.append(X_min)
                            start_idx_temp -= self.step_size
                            self.x_datasets.append(dataset)
                        
                        else:
                            if (start_idx_temp - n_samples + 1 >= 0):
                                a_idx = (start_idx_temp - n_samples).astype(int)
                                b_idx = (start_idx_temp + n_samples + 1).astype(int)
                                dataset.append(X_maj[a_idx:b_idx])
                                dataset.append(X_min)
                                start_idx_temp -= self.step_size
                                self.x_datasets.append(dataset)
                
                else:
                    cal_dis = len(majority_pseudo_sequence) - start_idx_temp - n_samples - 1
                    while (cal_dis >= 0):
                        dataset = []
                        a_idx = (start_idx_temp - n_samples).astype(int)
                        b_idx = (start_idx_temp + n_samples + 1).astype(int)
                        dataset.append(X_maj[a_idx:b_idx])
                        dataset.append(X_min)
                        start_idx_temp += self.step_size
                        cal_dis -= self.step_size
                        self.x_datasets.append(dataset)
                    start_idx_temp = start_idx
                    while (start_idx_temp - n_samples + 1 >= 0):
                        dataset = []
                        a_idx = (start_idx_temp - n_samples).astype(int)
                        b_idx = (start_idx_temp + n_samples + 1).astype(int)
                        dataset.append(X_maj[a_idx:b_idx])
                        dataset.append(X_min)
                        start_idx_temp -= self.step_size
                        self.x_datasets.append(dataset)

            else:
                if (start_idx + 1 < n_samples):
                    while (start_idx_temp + 1 + n_samples <= len(majority_pseudo_sequence)):
                        dataset = []
                        if (start_idx_temp + 1 <= n_samples):
                            a_idx = (start_idx_temp + n_samples + 1).astype(int)
                            dataset.append(X_maj[:a_idx])
                            dataset.append(X_min)
                            start_idx_temp += self.step_size
                            self.x_datasets.append(dataset)
                        
                        else:
                            if (start_idx_temp + 1 <= len(majority_pseudo_sequence)):
                                a_idx = (start_idx_temp - n_samples).astype(int)
                                b_idx = (start_idx_temp + n_samples + 1).astype(int)
                                dataset.append(X_maj[a_idx:b_idx])
                                dataset.append(X_min)
                                start_idx_temp += self.step_size
                                self.x_datasets.append(dataset)
                
                else:
                    cal_dis = start_idx_temp + 1 - n_samples
                    while (cal_dis >= 0):
                        dataset = []
                        a_idx = (start_idx_temp - n_samples).astype(int)
                        b_idx = (start_idx_temp + n_samples + 1).astype(int)
                        dataset.append(X_maj[a_idx:b_idx])
                        dataset.append(X_min)
                        start_idx_temp -= self.step_size
                        cal_dis -= self.step_size
                        self.x_datasets.append(dataset)
                    start_idx_temp = start_idx
                    while (start_idx_temp + n_samples + 1 <= len(majority_pseudo_sequence)):
                        dataset = []
                        a_idx = (start_idx_temp - n_samples).astype(int)
                        b_idx = (start_idx_temp + n_samples + 1).astype(int)
                        dataset.append(X_maj[a_idx:b_idx])
                        dataset.append(X_min)
                        start_idx_temp += self.step_size
                        self.x_datasets.append(dataset)

    def generate_base_classifiers(self):

        for i in range(len(self.x_datasets) - 1):
            X_maj, X_min = self.x_datasets[i][0], self.x_datasets[i][1]
            X = np.vstack([X_maj, X_min])
            Y = np.concatenate([np.zeros(len(X_maj), dtype=int), np.ones(len(X_min), dtype=int)])
            classifier = self.base_classifier
            classifier.fit(X, Y)
            self.classifiers.append(classifier)

    def select_classifiers(self, X_test):

        if len(self.classifiers) <= self.n_classifiers:
            return self.classifiers

        mean_test = np.mean(X_test, axis=0)
        mean_dis = []
        for i in range(len(self.x_datasets) - 1):
            mean_train = np.mean(np.concatenate(self.x_datasets[i]))
            mean_dis_temp = np.sum(np.square(mean_test - mean_train))
            mean_dis.append(mean_dis_temp)

        sorted_indices = np.argsort(mean_dis)
        chosen_classifiers = np.array(self.classifiers)[sorted_indices]
        return chosen_classifiers[:self.n_classifiers]

    def fit_and_predict(self, X_train, Y_train, X_test):

        X_maj = X_train[Y_train == 0]
        X_min = X_train[Y_train == 1]

        self.generate_datasets(X_maj, X_min)
        self.generate_base_classifiers()
        chosen_classifiers = self.select_classifiers(X_test)
        predictions = np.array([clf.predict(X_test) for clf in chosen_classifiers])
        majority_vote = np.round(predictions.mean(axis=0))

        return majority_vote

#testy
file_path = "C:/Users/Lenovo/Desktop/PCOS_data_without_infertility.xlsx"
X, y = preprocess_data(file_path)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=5)

X_maj = X_train[y_train == 0]
X_min = X_train[y_train == 1]

swsel_model = SWSEL(base_classifier=DecisionTreeClassifier(max_depth=3), step_size=7, n_classifiers=75)
y_predict = swsel_model.fit_and_predict(X_train, y_train, X_test)
print(accuracy_score(y_test, y_predict))
print(precision_score(y_test, y_predict))

# initial pos > 173 - błąd
# initial pos < 113 - okej