import numpy as np
from tabulate import tabulate

from scipy.stats import shapiro, ttest_rel, wilcoxon


def tables_one_param(method_name, param_list, table_style):

    pre_scores = np.load(f"wyniki/{method_name.lower()}_precision.npy")
    rec_scores = np.load(f"wyniki/{method_name.lower()}_recall.npy")
    f1_scores = np.load(f"wyniki/{method_name.lower()}_f1_score.npy")
    gm_scores = np.load(f"wyniki/{method_name.lower()}_g-mean.npy")

    scr = {"Precision":pre_scores, "Recall":rec_scores, "F1 score":f1_scores, 'G-mean':gm_scores}
    mean_scores = []
    std_scores = []
    for s in scr.keys():
        mean_scores.append(np.mean(scr[s], axis=1))
        std_scores.append(np.std(scr[s], axis=1))

    print(f"\n", f"Mean for {method_name}".center(50))
    table = tabulate(mean_scores, 
                    tablefmt=table_style, 
                    headers=param_list, 
                    showindex=["Precision", "Recall", "F1 score", 'G-mean']
    )
    print(table)

    print(f"\n", f"STD for {method_name}".center(60))
    table = tabulate(std_scores, 
                    tablefmt=table_style, 
                    headers=param_list, 
                    showindex=["Precision", "Recall", "F1 score", 'G-mean']
    )
    print(table)

def tables_two_params(method_name, first_param_list, second_param_list, table_style):
    
    pre_scores = np.load(f"wyniki/{method_name.lower()}_precision.npy")
    rec_scores = np.load(f"wyniki/{method_name.lower()}_recall.npy")
    f1_scores = np.load(f"wyniki/{method_name.lower()}_f1_score.npy")
    gm_scores = np.load(f"wyniki/{method_name.lower()}_g-mean.npy")

    scr = {"Precision":pre_scores, "Recall":rec_scores, "F1 score":f1_scores, 'G-mean':gm_scores}
    for s in scr.keys():
        means = np.mean(scr[s], axis=1)
        stds = np.std(scr[s], axis=1)
        mean_scores = []
        std_scores = []
        print(f"\n", f"Mean {s} dla {method_name}".center(50))
        for idx in range(len(first_param_list)):
            mean_scores.append(means[idx*len(second_param_list):(idx+1)*len(second_param_list)])
            std_scores.append(stds[idx*len(second_param_list):(idx+1)*len(second_param_list)])

        table = tabulate(mean_scores, 
                        tablefmt=table_style, 
                        headers=second_param_list, 
                        showindex=first_param_list
        )
        print(table)

        print(f"\n", f"STD {s} dla {method_name}".center(60))
        table = tabulate(std_scores, 
                        tablefmt=table_style, 
                        headers=second_param_list, 
                        showindex=first_param_list
        )
        print(table)

def tables_three_params(method_name, first_param_list, second_param_list, third_param_list, table_style):

    pre_scores = np.load(f"wyniki/{method_name.lower()}_precision.npy")
    rec_scores = np.load(f"wyniki/{method_name.lower()}_recall.npy")
    f1_scores = np.load(f"wyniki/{method_name.lower()}_f1_score.npy")
    gm_scores = np.load(f"wyniki/{method_name.lower()}_g-mean.npy")

    scr = {"Precision":pre_scores, "Recall":rec_scores, "F1 score":f1_scores, 'G-mean':gm_scores}
    indexes = [f"{x}-{y}" for x in second_param_list for y in third_param_list]
    for s in scr.keys():
        means = np.mean(scr[s], axis=2)
        stds = np.std(scr[s], axis=2)
        mean_scores = []
        std_scores = []
        for idx_1 in range(len(second_param_list)*len(third_param_list)):
            mean_scores.append(means[0][idx_1*len(first_param_list):(idx_1+1)*len(first_param_list)])   
            std_scores.append(stds[0][idx_1*len(first_param_list):(idx_1+1)*len(first_param_list)])       

        print(f"\n", f"Mean {s} dla {method_name}".center(60))
        table = tabulate(mean_scores, 
                        tablefmt=table_style, 
                        headers=first_param_list, 
                        showindex=indexes
        )
        print(table)

        print(f"\n", f"STD {s} dla {method_name}".center(60))
        table = tabulate(std_scores, 
                        tablefmt=table_style, 
                        headers=first_param_list, 
                        showindex=indexes
        )
        print(table)


M_list = [25, 50, 75, 100]
lr_list = [0.1, 0.5, 1, 10]
alpha_list=[0.1 * i for i in range(1, 10)]
# tables_one_param("SWSEL", M_list, "grid")
# tables_one_param("SMRF", M_list, "grid")
# tables_two_params("SMAB",  M_list, lr_list, "grid")
# tables_three_params("VASA", alpha_list, M_list, lr_list, "grid")

def compare_models(scores, model_names, table_style="grid", alpha=0.05, alternative="two-sided"):
    stat_matrix = [[None for _ in range(scores.shape[0])] for _ in range(scores.shape[0])]
    for i in range(scores.shape[0]):
        for j in range(scores.shape[0]):
            t1, p1 = shapiro(scores[i])
            t2, p2 = shapiro(scores[j])
            if p1 > alpha and p2 > alpha:
                t, p = ttest_rel(scores[i], scores[j], alternative=alternative)
                stat_matrix[i][j] = f"t, {p:.4f}"
            else:
                t, p = wilcoxon(scores[i], scores[j], alternative=alternative)
                stat_matrix[i][j] = f"w, {p:.4f}"

    table = tabulate(stat_matrix,
                    tablefmt=table_style, 
                    headers=model_names, 
                    showindex=model_names)
    
    print("\nSignificance Matrix (1 = significant difference, 0 = no significant difference):")
    print(table)
    return table

#For comparing form .npy file, scores should be extracted from a file
method_name = "rusab"
metric = "Precision"
alpha = 0.05
alternative_hypothesis = "greater"
scores = np.load(f"wyniki/{method_name.lower()}_{metric.lower()}.npy")
if method_name == "VASA":
    scores = scores[0]

vasa_names = [f"M {x}-lr {y}-a {z}" for x in M_list for y in lr_list for z in alpha_list]
rusab_names = [f"M {x}- lr {y}" for x in M_list for y in lr_list]
smrf_names = M_list

compare_models(scores, rusab_names, "latex")
