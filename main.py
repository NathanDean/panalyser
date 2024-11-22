## Executes imputation

import pandas as pd
from panalyser import optimise_imputation, run_imputation

df = pd.read_csv('./data/test_data.csv')
data_cols = ['Under-5 mortality rate - All', 'No. of under-5 deaths - All', 'Early breastfeeding rate', 'Exclusive breastfeeding rate']
ts = 'Year'
cs = 'Area'
bounds = '''matrix(c(
        3, 0, 35,
        4, 0, 385000,
        5, 0, 95,
        6, 0, 90
    ), ncol = 3, byrow = TRUE)'''
m = 20
n_calls = 50

params, worst_score, final_score = optimise_imputation(df, data_cols, m, n_calls, ts, cs, bounds)
imputed_dfs = run_imputation(df, data_cols, m, params, ts, cs, bounds)

for i, df in enumerate(imputed_dfs):

    df.to_csv(f'./data/imputed_data/imputed_df_{i + 1}.csv', index = False)