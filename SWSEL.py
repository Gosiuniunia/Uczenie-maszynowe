import numpy as np

class SWSEL:
    def __init__(self, estimator, n_estimators, step_size=1):

        self.estimator = estimator
        self.step_size = step_size
        self.n_estimators = n_estimators
        self.x_datasets = []
        self.classifiers = []
        self.chosen_classifiers = []

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
                "punkt początkowy jest za połową sekwencji"
                if (len(majority_pseudo_sequence) - start_idx < n_samples):
                    "punkt początkowy jest bliżej końca niż wielkość połowy okna"
                    while (start_idx_temp - n_samples >= 0):
                        "dopóki możemy się przesuwać w lewo"
                        dataset = []
                        if (len(majority_pseudo_sequence) - start_idx_temp <= n_samples):
                            "przesuwamy się do momentu aż liczba instancji po prawej będzie większa równa wielkości X_min"
                            print('a0', start_idx_temp - n_samples)
                            dataset.append(X_maj[a_idx:])
                            dataset.append(X_min)
                            start_idx_temp -= self.step_size
                            self.x_datasets.append(dataset)
                        
                        else:
                            a_idx = (start_idx_temp - n_samples).astype(int)
                            b_idx = (start_idx_temp + n_samples).astype(int)
                            dataset.append(X_maj[a_idx:b_idx])
                            dataset.append(X_min)
                            start_idx_temp -= self.step_size
                            self.x_datasets.append(dataset)
                
                else:
                    cal_dis = len(majority_pseudo_sequence) - start_idx_temp - n_samples
                    while (cal_dis >= 0):
                        "przesuwamy się w prawo do końca dopóki odl jest równa połowie wielkości okna"
                        dataset = []
                        a_idx = (start_idx_temp - n_samples).astype(int)
                        b_idx = (start_idx_temp + n_samples).astype(int)
                        dataset.append(X_maj[a_idx:b_idx])
                        dataset.append(X_min)
                        start_idx_temp += self.step_size
                        cal_dis -= self.step_size
                        self.x_datasets.append(dataset)
                    start_idx_temp = start_idx
                    "cofamy się do pozycji początkowej"
                    while (start_idx_temp - n_samples >= 0):
                        "przesuwamy się w lewo do końca jeśli krok pozwoli"
                        dataset = []
                        a_idx = (start_idx_temp - n_samples).astype(int)
                        b_idx = (start_idx_temp + n_samples).astype(int)
                        dataset.append(X_maj[a_idx:b_idx])
                        dataset.append(X_min)
                        start_idx_temp -= self.step_size
                        self.x_datasets.append(dataset)

            else:
                "jesli punkt początkowy jest na 1 połowie"
                if (start_idx + 1 < n_samples):
                    "jesli po lewo jest mniej niż połowa okna"
                    while (start_idx_temp + n_samples <= len(majority_pseudo_sequence)):
                        "dopóki można się przesuwać w prawo"
                        dataset = []
                        if (start_idx_temp + 1 <= n_samples):
                            "tworzenie danych z początku datasetu do momentu aż wielkość od początku będzie równa wielkości okna"
                            a_idx = (start_idx_temp + n_samples).astype(int)
                            dataset.append(X_maj[:a_idx])
                            dataset.append(X_min)
                            start_idx_temp += self.step_size
                            self.x_datasets.append(dataset)
                        
                        else:
                            "tworzenie danych do końca datasetu"
                            a_idx = (start_idx_temp - n_samples).astype(int)
                            b_idx = (start_idx_temp + n_samples).astype(int)
                            print('ab3', start_idx_temp - n_samples, start_idx_temp + n_samples)
                            dataset.append(X_maj[a_idx:b_idx])
                            print(len(X_maj[a_idx:b_idx]))
                            dataset.append(X_min)
                            start_idx_temp += self.step_size
                            self.x_datasets.append(dataset)
                
                else:
                    cal_dis = start_idx_temp - n_samples
                    while (cal_dis >= 0):
                        dataset = []
                        a_idx = (start_idx_temp - n_samples).astype(int)
                        b_idx = (start_idx_temp + n_samples).astype(int)
                        dataset.append(X_maj[a_idx:b_idx])
                        dataset.append(X_min)
                        start_idx_temp -= self.step_size
                        cal_dis -= self.step_size
                        self.x_datasets.append(dataset)
                    start_idx_temp = start_idx
                    while (start_idx_temp + n_samples <= len(majority_pseudo_sequence)):
                        dataset = []
                        a_idx = (start_idx_temp - n_samples).astype(int)
                        b_idx = (start_idx_temp + n_samples).astype(int)
                        dataset.append(X_maj[a_idx:b_idx])
                        dataset.append(X_min)
                        start_idx_temp += self.step_size
                        self.x_datasets.append(dataset)
        else:
            "analogicznie dla nieparzystej liczby danych"
            n_samples = (window_size - 1) / 2

            if (start_idx > len(majority_pseudo_sequence) / 2):
                if (len(majority_pseudo_sequence) - start_idx - 1 < n_samples):
                    while (start_idx_temp - n_samples >= 0):
                        dataset = []
                        if (len(majority_pseudo_sequence) - start_idx_temp <= n_samples):
                            a_idx = (start_idx_temp - n_samples).astype(int)
                            dataset.append(X_maj[a_idx:]) 
                            dataset.append(X_min)
                            start_idx_temp -= self.step_size
                            self.x_datasets.append(dataset)
                        
                        else:
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
                    while (start_idx_temp - n_samples >= 0):
                        dataset = []
                        a_idx = (start_idx_temp - n_samples).astype(int)
                        b_idx = (start_idx_temp + n_samples + 1).astype(int)
                        dataset.append(X_maj[a_idx:b_idx])
                        dataset.append(X_min)
                        start_idx_temp -= self.step_size
                        self.x_datasets.append(dataset)

            else:
                if (start_idx + 1 < n_samples):
                    while (start_idx_temp + n_samples <= len(majority_pseudo_sequence)):
                        dataset = []
                        if (start_idx_temp <= n_samples):
                            a_idx = (start_idx_temp + n_samples).astype(int)
                            dataset.append(X_maj[:a_idx])
                            dataset.append(X_min)
                            start_idx_temp += self.step_size
                            self.x_datasets.append(dataset)
                        
                        else:
                            a_idx = (start_idx_temp - n_samples - 1).astype(int)
                            b_idx = (start_idx_temp + n_samples).astype(int)
                            dataset.append(X_maj[a_idx:b_idx])
                            dataset.append(X_min)
                            start_idx_temp += self.step_size
                            self.x_datasets.append(dataset)
                
                else:
                    cal_dis = start_idx_temp - n_samples
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

        for i in range(len(self.x_datasets)):
            X_maj, X_min = self.x_datasets[i][0], self.x_datasets[i][1]
            X = np.vstack([X_maj, X_min])
            Y = np.concatenate([np.zeros(len(X_maj), dtype=int), np.ones(len(X_min), dtype=int)])
            classifier = self.estimator
            classifier.fit(X, Y)
            self.classifiers.append(classifier)

    def select_classifiers(self, X_test):

        if len(self.classifiers) <= self.n_estimators:
            return self.classifiers

        mean_test = np.mean(X_test, axis=0)
        mean_dis = []
        for i in range(len(self.x_datasets)):
            mean_train = np.mean(np.concatenate(self.x_datasets[i]))
            mean_dis_temp = np.sum(np.square(mean_test - mean_train))
            mean_dis.append(mean_dis_temp)

        sorted_indices = np.argsort(mean_dis)
        chosen_classifiers = np.array(self.classifiers)[sorted_indices]
        return chosen_classifiers[:self.n_estimators]
    
    def fit(self, X_train, Y_train):
        X_maj = X_train[Y_train == 0]
        X_min = X_train[Y_train == 1]

        self.generate_datasets(X_maj, X_min)
        self.generate_base_classifiers()

    def predict(self, X_test):
        self.chosen_classifiers = self.select_classifiers(X_test)
        predictions = np.array([clf.predict(X_test) for clf in self.chosen_classifiers])
        majority_vote = np.round(predictions.mean(axis=0))

        return majority_vote
