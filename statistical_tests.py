import numpy as np
from tabulate import tabulate

from scipy.stats import shapiro, ttest_rel, wilcoxon


def generate_model_names(classifier_name, oversampling_name):
    M_list = [25, 50, 75, 100]
    lr_list = [0.1, 0.5, 1, 10]
    alpha_list = [round(0.1 * i,1) for i in range(1, 11)]

    # A list of the parameters permutation - models names, used as column headers.
    if oversampling_name.lower() == "vao":
        if classifier_name.lower() == "ab" or classifier_name.lower() == "sab":
            model_names =  [f"{x}-{y}-{z}" for x in M_list for y in lr_list for z in alpha_list]
        else:
            model_names = [f"{x}-{y}" for x in M_list for y in alpha_list]
    else:
        if classifier_name.lower() == "ab" or classifier_name.lower() == "sab":
            model_names = [f"{x}-{y}" for x in M_list for y in lr_list]
        else:
            model_names = M_list

    return model_names

def print_scores(classifier_name, oversampling_name, rounding=None, table_style="grid", T=False):
    
    model_names = generate_model_names(classifier_name, oversampling_name)
    metrics = ["Precision", "Recall", "F1 score", "G-mean"]

    pre_scores = np.load(f"wyniki/{classifier_name.lower()}_{oversampling_name.lower()}_precision.npy")
    rec_scores = np.load(f"wyniki/{classifier_name.lower()}_{oversampling_name.lower()}_recall.npy")
    f1_scores = np.load(f"wyniki/{classifier_name.lower()}_{oversampling_name.lower()}_f1_score.npy")
    gm_scores = np.load(f"wyniki/{classifier_name.lower()}_{oversampling_name.lower()}_g-mean.npy")

    scr = {"Precision":pre_scores, "Recall":rec_scores, "F1 score":f1_scores, "G-mean":gm_scores}
    mean_scores = []
    std_scores = []
    for s in scr.keys():
        if rounding != None:
            mean_scores.append(np.round(np.mean(scr[s], axis=1), rounding))
            std_scores.append(np.round(np.std(scr[s], axis=1), rounding))
        else:
            mean_scores.append(np.mean(scr[s], axis=1))
            std_scores.append(np.std(scr[s], axis=1))

    mean_scores = np.array(mean_scores)
    std_scores = np.array(std_scores)
    scores = np.char.add(np.char.add(mean_scores.astype(str), u' \u00B1 '), std_scores.astype(str))
    if T == True:
        scores_T = scores.T
        table = tabulate(scores_T, 
                        tablefmt=table_style, 
                        headers=metrics, 
                        showindex=model_names
        )
    elif T == False:
        table = tabulate(scores, 
                        tablefmt=table_style, 
                        headers=model_names, 
                        showindex=metrics
        )

    if table_style == "grid":
        print(f"\n", f"Scores for {classifier_name} classifiers with {oversampling_name} oversampling")
        print(table)
    else:
        table_latex = table[:-13] + f"\caption{{Scores for {classifier_name} classifier with {oversampling_name} oversampling}}\n" + table[-13:]
        print(table_latex, "\n")


# print_scores("ab", "vao", rounding=3, T=True)

def compare_models(scores, model_names, table_style="grid", alpha=0.05, alternative="two-sided"):
    stat_matrix = [[None for _ in range(scores.shape[0])] for _ in range(scores.shape[0])]
    for i in range(scores.shape[0]):
        for j in range(scores.shape[0]):
            if i == j: #comparison with oneself is omitted
                stat_matrix[i][j] = "nan"
                continue
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
    
    if table_style == "grid":
        print("\n Matrix of p-values from paired statistical tests between models")
        print(table)
    else:
        table_latex = table[:-13] + f"\caption{{Matrix of p-values from paired statistical tests between models}}\n" + table[-13:]
        print(table_latex)

#For comparing form .npy file, scores should be extracted from a file

classifier_name = 'ab'
oversampling_name = 'none'
metric_name = "precision"
scores = np.load(f"wyniki/{classifier_name.lower()}_{oversampling_name.lower()}_{metric_name}.npy")
model_names = generate_model_names(classifier_name, oversampling_name)
compare_models(scores, model_names, "latex")
