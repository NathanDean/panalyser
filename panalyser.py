import os
os.environ['R_HOME'] = r'C:\Program Files\R\R-4.4.1'

import pandas as pd
import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer

from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

amelia = importr('Amelia')
pandas2ri.activate()

data_cols = ['Under-5 mortality rate - All', 'No. of under-5 deaths - All', 'Early breastfeeding rate', 'Exclusive breastfeeding rate']

def clean_numeric_columns(df, data_cols):

    df_clean = df.copy()
    prob_cols = []

    for col in data_cols:

        try:

            df_clean[col] = df_clean[col].astype(str).str.strip()
            df_clean[col] = df_clean[col].replace(r'[^0-9.-]', '', regex = True)

            numeric_col = pd.to_numeric(df_clean[col], errors = 'coerce')
            df_clean[col] = numeric_col

            percentage_null = numeric_col.isna().mean() * 100

            if percentage_null > 10:

                prob_cols.append({

                    'column': col,
                    'percentage_null': percentage_null,
                    'sample': df_clean[col].head().tolist()

                })

        except Exception as e:

            prob_cols.append({

                'column': col,
                'error': str(e),
                'sample': df_clean[col].head().tolist()

            })

    return df_clean, prob_cols

def run_imputation(df, m, params):

    df_clean, prob_cols = clean_numeric_columns(df, data_cols)

    if prob_cols:

        print('Warning: Problems in following columns:')

        for col in prob_cols:

            print(f'Column: {col}')
            print(f'Sample values: {col['sample']}')

            if 'percentage_null' in col:

                print(f'Percentage null: {col['percentage_null']}')

            if 'error' in col:

                print(f'Error: {col['error']}')

        print('Continuing imputation with cleaned data')
    
    r_df = pandas2ri.py2rpy(df)

    for col in data_cols:

        if not pd.api.types.is_numeric_dtype(df_clean[col]):

            raise ValueError(f'{col} is not numeric after cleaning')
    
    bounds = robjects.r('''matrix(c(
        3, 0, 35,
        4, 0, 385000,
        5, 0, 95,
        6, 0, 90
    ), ncol = 3, byrow = TRUE)''')

    results = amelia.amelia(
        r_df,
        m = m,
        ts = 'Year',
        cs = 'Area',
        bounds = bounds,
        tolerance = params[0],
        empri = params[1],
        max_resample = params[2],
    )

    imputed_dfs = []

    for i in range(m):

        try:

            imputed_data = np.array(results.rx2('imputations')[i])
            imputed_data_shape = imputed_data.shape
            expected_data_shape = (len(df), len(df.columns))

            if imputed_data_shape != expected_data_shape:

                if imputed_data_shape[0] == expected_data_shape[1] and imputed_data_shape[1] == expected_data_shape[0]:

                    imputed_data = imputed_data.T

                else:

                    raise ValueError(f'Error: Expected data shape {expected_data_shape}, got {imputed_data_shape}')
                
            imputed_df = pd.DataFrame(imputed_data, columns = df.columns, index = df.index)

            for col in data_cols:

                imputed_df[col] = pd.to_numeric(imputed_df[col], errors = 'raise')

                if not pd.api.types.is_numeric_dtype(imputed_df[col]):

                    print(f'Non-numeric values found in {col}')
                    print(imputed_df[col].head())
                    raise ValueError(f'{col} is non-numeric after imputation')

            imputed_dfs.append(imputed_df)

        except Exception as e:

            print(f'Error in DataFrame construction: {e}')

            if imputed_data.shape [0] == len(df.columns):

                print(f'Error during imputation {i + 1}: {e}')

    return imputed_dfs


def evaluate_imputation(imputed_dfs, original_stats):

    original_correlations = original_stats['correlations']
    original_means = original_stats['means']
    original_variances = original_stats['variances']

    scores = []
    
    for df in imputed_dfs:

        try:
        
            imp_correlations, imp_means, imp_variances = df[data_cols].corr().values, df.mean().values(), df.var().values()

            if(

                np.isnan(imp_correlations).any() or
                np.isnan(imp_means).any() or
                np.isnan(imp_variances).any()

            ):
                
                scores.append(float('inf'))
                continue

            correlations_error = np.mean(np.abs(original_correlations - imp_correlations))
            means_error = np.mean(np.abs(original_means - imp_means))
            variances_error = np.mean(np.abs(original_variances - imp_variances))

            imputation_score = np.mean([correlations_error, means_error, variances_error])

            scores.append(imputation_score)
        
        except Exception as e:

            print(f'Error during evaluation: {e}')
            scores.append(float('inf'))
        
    if not scores or all(np.isinf(scores)):

        return float('inf')
    
    overall_score = np.mean([s for s in scores if not np.isinf(s)])
    return overall_score


def objective_function(df, original_stats, m, params):

    try:
        
        imputed_dfs = run_imputation(df, m, params)
        score = evaluate_imputation(imputed_dfs, original_stats)

        if np.isinf(score):

            return 1e10

        return score

    except Exception as e:
        
        print(f'Error during optimisation: {e}')
        return float('1e10')


def optimise_imputation(df, m, n_calls):
    
    n_observations = len(df)
    
    original_stats = {

        'correlations': df[data_cols].corr().values,
        'means': df[data_cols].mean(),
        'variances': df[data_cols].var()
    
    }

    if (np.isnan(original_stats['correlations']).any()
        or np.isnan(original_stats['means']).any()
        or np.isnany(original_stats['variances']).any()):

            print('Warning: Original correlations, means or variances are NaN')

    params = [

        Real(1e-5, 1e-1, name = 'tolerance'),
        Real(n_observations * 0.005, n_observations * 0.01, name = 'empri'),
        Integer(1, 1000, name = 'max_resample')

    ]

    evaluation_scores = []
    
    def wrapped_objective_function(params):

        score = objective_function(df, original_stats, m, params)
        
        if not np.isinf(score):
            evaluation_scores.append(score)
        
        return score

    result = gp_minimize(

        wrapped_objective_function,
        params,
        n_calls = n_calls,
        n_random_starts = 5,
        acq_func = 'EI'

    )

    optimal_parameters = [
    
        result.x[0],        # tolerance   
        result.x[1],        # empri
        int(result.x[2])    # max_resample
    
    ]

    worst_score = max(evaluation_scores) if evaluation_scores else float('inf')
    final_score = result.fun

    return optimal_parameters, worst_score, final_score