import numpy as np
from tabulate import tabulate
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import shapiro, ttest_rel, wilcoxon


def generate_model_names(classifier_name, oversampling_name):
    M_list = [25, 50, 75, 100]
    lr_list = [0.1, 0.5, 1, 10]
    alpha_list = [round(0.1 * i,1) for i in range(1, 11)]

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

def generate_dataframe(model_names, scores):
    first_parts = model_names[0].split("-")
    num_parts = len(first_parts)

    M_vals, lr_vals, alpha_vals = [], [], []
    ordered_cols = ["M"]

    for name in model_names:
        parts = name.split("-")
        M_vals.append(int(parts[0]))

        if num_parts >= 2:
            lr_vals.append(float(parts[1]))
        if num_parts == 3:
            alpha_vals.append(float(parts[2]))

    data = {
        "M": M_vals,
        "Precision": scores[0],
        "Recall": scores[1],
        "F1": scores[2],
        "Gmean": scores[3]
    }

    if num_parts >= 2:
        data["lr"] = lr_vals
        ordered_cols.append("lr")
    if num_parts == 3:
        data["alpha"] = alpha_vals
        ordered_cols.append("alpha")

    df = pd.DataFrame(data)
    metric_cols = [col for col in df.columns if col not in ordered_cols]
    df = df[ordered_cols + metric_cols]

    return df

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
        print(f"\n", f"Wyniki klasyfikatorów {classifier_name} z samplingiem {oversampling_name}")
        print(table)
    else:
        table_latex = "\\begin{table}[H]\n\centering"+ table + f"\n\caption{{Wyniki klasyfikatorów {classifier_name} z samplingiem {oversampling_name}}}\n\end{{table}}\n"
        print(table_latex, "\n")
        return table_latex

    df_mean = generate_dataframe(model_names, mean_scores)
    df_std = generate_dataframe(model_names, std_scores)
    return df_mean, df_std


def compare_models(scores, model_names, table_style="grid", alpha=0.05, alternative="two-sided"):
    stat_matrix = [[None for _ in range(scores.shape[0])] for _ in range(scores.shape[0])]
    for i in range(scores.shape[0]):
        for j in range(scores.shape[0]):
            if i <= j: #comparison with oneself and double testing is omitted
                stat_matrix[i][j] = "-"
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
        print("\n Macierz wartości p dla testów statystycznych parami między modelami")
        print(table)
    else:
        table_latex = "\\begin{table}[H]\n\centering"+ table + "\caption{Macierz wartości p dla testów statystycznych parami między modelami}\n\end{table}\n"
        print(table_latex)
        return table_latex

# classifiers = ["SWSEL", "RF", "AB", "SAB"]
# oversamplings = ["NONE", "SMOTE", "RUS", "VAO"]
# file = "tables.txt"

# Table generation
# with open(file, "w", encoding="utf-8") as f:
#     for clf in classifiers:
#         f.write(f"\subsection{{Wyniki dla klasyfikatora {clf} dla różnych over-samplingów}}")
#         for over in oversamplings:
#             result = print_scores(clf, over, rounding=3, table_style="latex", T=True)
#             f.write(result)
#             f.write("\n\n")

# data = {}
# best_params_num = [0, 0, 0, 6, 2, 1, 1, 12, 8, 8, 4, 40, 0, 0, 8, 64]
# i = 0
# for clf in classifiers:
#     for over in oversamplings:
#         pre_scores = np.load(f"wyniki/{clf.lower()}_{over.lower()}_precision.npy")[best_params_num[i]]
#         rec_scores = np.load(f"wyniki/{clf.lower()}_{over.lower()}_recall.npy")[best_params_num[i]]
#         f1_scores = np.load(f"wyniki/{clf.lower()}_{over.lower()}_f1_score.npy")[best_params_num[i]]
#         gm_scores = np.load(f"wyniki/{clf.lower()}_{over.lower()}_g-mean.npy")[best_params_num[i]]

#         scr = {"Precision":pre_scores, "Recall":rec_scores, "F1 score":f1_scores, "G-mean":gm_scores}
#         data[f"{clf}_{over}"] = scr
#         i += 1

# file = "compare.txt"

# Test grouped by classificator
# part_data = [[], [], [], []]
# i = 0
# for part in classifiers:
#     for key, value in data.items():
#         if part == "AB":
#             if part in key and "SAB" not in key:
#                 part_data[i].extend(value["Precision"])
#         else:
#             if part in key:
#                 part_data[i].extend(value["Precision"])

#     i += 1

# with open(file, "w", encoding="utf-8") as f:
#     f.write(compare_models(np.array(part_data)[:3], classifiers[:3], table_style="latex", alternative="greater"))


# Test grouped by oversampling
# part_data = [[], [], [], []]
# i = 0
# for part in oversamplings:
#     for key, value in data.items():
#         if part in key and "SAB" not in key:
#             print(key)
#             part_data[i].extend(value["Precision"])
#     i += 1

# print(len(part_data[0]))
# with open(file, "w", encoding="utf-8") as f:
    # f.write(compare_models(np.array(part_data), oversamplings, table_style="latex", alternative="greater"))


# Test for all combinations
# metric = "Precision"
# model_names = [key for key in data.keys()]
# model_names = [key for key in data.keys() if "SWSEL" not in key]
# scores = np.array([data[key][metric] for key in data])
# scores = np.array([data[key][metric] for key in data if "SWSEL" not in key])

# with open(file, "w", encoding="utf-8") as f:
    # f.write(compare_models(scores, model_names, table_style="latex"))

# AB and SAB hyperparamethers testing visualization without VAO
# fig, ax = plt.subplots(1, 3, figsize=(15, 4))

# i = 0
# for over in oversamplings[:3]:
#     df_ab, _ = print_scores("AB", over, rounding=3)
#     df_sab, _ = print_scores("SAB", over, rounding=3)
#     labels = generate_model_names("AB", over)
#     colors = ['orchid', 'teal']
#     ax[i].plot(labels, list(df_ab["Precision"]), colors[0], label="AB")
#     ax[i].plot(labels, list(df_sab["Precision"]), colors[1], label="SAB")
#     ax[i].legend(title='Klasyfikator')
#     ax[i].set_title(f'{over}')
#     ax[i].set_xticklabels(labels, rotation=90)
#     i += 1

# VAO visualization
# file = "vao.jpg"
# fig, ax = plt.subplots(figsize=(30, 4))
# df_ab, _ = print_scores("AB", "VAO", rounding=3)
# df_sab, _ = print_scores("SAB", "VAO", rounding=3)
# labels = generate_model_names("AB", "VAO")
# colors = ['orchid', 'teal']
# ax.plot(labels, list(df_ab["Precision"]), colors[0], label="AB")
# ax.plot(labels, list(df_sab["Precision"]), colors[1], label="SAB")
# ax.legend(title='Klasyfikator')
# ax.set_title(f'{over}')
# ax.set_xticklabels(labels, rotation=90)

# All combination's metrics visualization
# file = "metrics.jpg"
# labels = list(data.keys())
# metrics = ['Precision', 'Recall', 'F1 score', 'G-mean']
# n_metrics = len(metrics)
# n_labels = len(labels)
# colors = ['orchid', 'mediumpurple', 'skyblue', 'teal']

# averaged_data = {}
# for key, metric_values in data.items():
#     averaged_data[key] = {metric: np.mean(values) for metric, values in metric_values.items()}

# df_avg = pd.DataFrame.from_dict(averaged_data, orient='index')

# fig, ax = plt.subplots(figsize=(16, 6)) 

# width = 0.15 

# x = np.arange(n_labels) 

# for i, metric in enumerate(metrics):
#     offset = width * (i - (n_metrics - 1) / 2)
#     ax.bar(x + offset, df_avg[metric], width, label=metric, color=colors[i % len(colors)])

# ax.set_ylabel('Uśredniona Wartość Metryki')
# ax.set_xticks(x)
# ax.set_xticklabels(labels, rotation=45, ha='right') 
# ax.legend(title='Metryka', loc='lower right')
# ax.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()


# plt.savefig(file, dpi=300, bbox_inches='tight')

# plt.show()