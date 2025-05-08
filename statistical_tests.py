import numpy as np
from tabulate import tabulate

from scipy.stats import shapiro, ttest_rel, wilcoxon

#Shapiro-Wilk, Wilcoxon, t-test
# tablefmt="latex"

def tables_one_param(method_name, param_list):
    pre_scores = np.load(f"{method_name.lower()}_precision.npy")
    rec_scores = np.load(f"{method_name.lower()}_recall.npy")
    f1_scores = np.load(f"{method_name.lower()}_f1_score.npy")
    gm_scores = np.load(f"{method_name.lower()}_g-mean.npy")

    print(f"\n", f"Wyniki średniej dla {method_name}".center(50))
    table = tabulate([np.mean(pre_scores, axis=1), np.mean(rec_scores, axis=1), np.mean(f1_scores, axis=1), np.mean(gm_scores, axis=1)], 
                    tablefmt="grid", 
                    headers=param_list, 
                    showindex=["Precision", "Recall", "F1 score", 'G-mean']
    )
    print(table)

    print(f"\n", f"Wyniki odchylenia standardowego dla {method_name}".center(60))
    table = tabulate([np.std(pre_scores, axis=1), np.std(rec_scores, axis=1), np.std(f1_scores, axis=1), np.std(gm_scores, axis=1)], 
                    tablefmt="grid", 
                    headers=param_list, 
                    showindex=["Precision", "Recall", "F1 score", 'G-mean']
    )
    print(table)

def tables_two_params(method_name, first_param_list, second_param_list):
    pre_scores = np.load(f"{method_name.lower()}_precision.npy")
    rec_scores = np.load(f"{method_name.lower()}_recall.npy")
    f1_scores = np.load(f"{method_name.lower()}_f1_score.npy")
    gm_scores = np.load(f"{method_name.lower()}_g-mean.npy")

    scr = {"Precision":pre_scores, "Recall":rec_scores, "F1 score":f1_scores, 'G-mean':gm_scores}
    for s in scr.keys():
        mean_scores = []
        std_scores = []
        print(f"\n", f"Uśredniony {s} dla {method_name}".center(50))
        for idx_1 in range(len(first_param_list)):
            row_means = []
            row_stds = []
            for idx_2 in range(len(second_param_list)):
                idx = idx_2 * 4 + idx_1
                row_means.append(np.mean(scr[s][idx]))   
                row_stds.append(np.std(scr[s][idx]))     
            mean_scores.append(row_means)
            std_scores.append(row_stds)

        table = tabulate(mean_scores, 
                        tablefmt="grid", 
                        headers=second_param_list, 
                        showindex=first_param_list
        )
        print(table)

        print(f"\n", f"Odchylenie standardowe {s} dla {method_name}".center(60))
        table = tabulate(std_scores, 
                        tablefmt="grid", 
                        headers=second_param_list, 
                        showindex=first_param_list
        )
        print(table)

def tables_three_params(method_name, first_param_list, second_param_list, third_param_list): #not yet done
    pre_scores = np.load(f"{method_name.lower()}_precision.npy")
    rec_scores = np.load(f"{method_name.lower()}_recall.npy")
    f1_scores = np.load(f"{method_name.lower()}_f1_score.npy")
    gm_scores = np.load(f"{method_name.lower()}_g-mean.npy")

    scr = {"Precision":pre_scores, "Recall":rec_scores, "F1 score":f1_scores, 'G-mean':gm_scores}
    for s in scr.keys():
        print(f"{s}")
        mean_scores = []
        std_scores = []
        for idx_1 in range(len(first_param_list)):
            print(f"\n", f"{first_param_list[idx_1]}".center(50))
            for idx_2 in range(len(second_param_list)):
                row_means = []
                row_stds = []
                for idx_3 in range(len(third_param_list)):
                    idx = idx_3 * 4 + idx_2
                    row_means.append(np.mean(scr[s][idx_1][idx]))   
                    row_stds.append(np.std(scr[s][idx_1][idx]))     
                mean_scores.append(row_means)
                std_scores.append(row_stds)

            table = tabulate(mean_scores, 
                            tablefmt="grid", 
                            headers=third_param_list, 
                            showindex=second_param_list
            )
            print(table)

            print(f"\n", f"Odchylenie standardowe {s} dla {method_name}".center(60))
            table = tabulate(std_scores, 
                            tablefmt="grid", 
                            headers=third_param_list, 
                            showindex=second_param_list
            )
            print(table)


tables_one_param("SWSEL", ["M 25", "M 50", "M 75", "M 100"])
tables_one_param("SMRF", ["M 25", "M 50", "M 75", "M 100"])
tables_two_params("SMAB", ["lr 0.1", "lr 0.5", "lr 1", "lr 10"], ["M 25", "M 50", "M 75", "M 100"])
# tables_three_params("VASA", ["M 25", "M 50", "M 75", "M 100"], ["lr 0.1", "lr 0.5", "lr 1", "lr 10"], [0.1 * i for i in range(1, 10)])
