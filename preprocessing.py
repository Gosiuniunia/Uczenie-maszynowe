import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif

def preprocess_data(file_path):

    excel_data = pd.read_excel(file_path, sheet_name=1)


    excel_data = excel_data.drop(columns=["Unnamed: 44", "Sl. No", "Patient File No."])

    empty_values_ff = excel_data[excel_data['Fast food (Y/N)'].isna()]
    excel_data.loc[empty_values_ff.index, 'Fast food (Y/N)'] = excel_data.loc[empty_values_ff.index, 'Fast food (Y/N)'].replace(np.nan, 0)

    empty_values_mm = excel_data[excel_data['Marraige Status (Yrs)'].isna()].index
    excel_data.loc[empty_values_mm, 'Marraige Status (Yrs)'] = excel_data.loc[empty_values_mm, 'Marraige Status (Yrs)'].replace(np.nan, 0)

    excel_data = excel_data[~excel_data['AMH(ng/mL)'].apply(lambda x: isinstance(x, str))]

    text_values_beta_hcg = excel_data[excel_data['II    beta-HCG(mIU/mL)'].apply(lambda x: isinstance(x, str))]
    excel_data.loc[text_values_beta_hcg.index, 'II    beta-HCG(mIU/mL)'] = excel_data['II    beta-HCG(mIU/mL)'].apply(lambda x: x[:-1] if isinstance(x, str) else x)

    excel_data['II    beta-HCG(mIU/mL)'] = pd.to_numeric(excel_data['II    beta-HCG(mIU/mL)'], errors='coerce')
    excel_data['AMH(ng/mL)'] = pd.to_numeric(excel_data['AMH(ng/mL)'], errors='coerce')

    X = excel_data.drop('PCOS (Y/N)', axis=1)
    y = excel_data['PCOS (Y/N)'] 

    mi = mutual_info_classif(X, y)
    mi_df = pd.DataFrame({'Feature': X.columns, 'Mutual Information': mi})

    mi_df = mi_df.sort_values(by='Mutual Information', ascending=False)

    top_15_columns = mi_df['Feature'].head(15)

    reduced_X = excel_data[top_15_columns]

    return reduced_X.to_numpy(), y.to_numpy()