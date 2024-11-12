import pandas as pd
from panalyser import optimise_imputation, run_imputation

df = pd.read_csv('./data/test_data.csv')
m = 20

# df = user input
# m = user input

params, worst_score, final_score = optimise_imputation(df, m, 50)
imputed_dfs = run_imputation(df, m, params)

for i, df in enumerate(imputed_dfs):

    df.to_csv(f'./data/imputed_data/imputed_df_{i + 1}')