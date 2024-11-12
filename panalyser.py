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

def run_imputation(df, m, params):

    df_clean = df.copy()

    for col in data_cols:

        try:

            df_clean[col] = pd.to_numeric(df_clean[col], errors = 'coerce')

        except Exception as e:

            print(f'Error converting {col} to numeric: {e}')
    
    r_df = pandas2ri.py2rpy(df)
    
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

    # for m in range(m):
    #     imputed_df = pd.DataFrame(np.array(results.rx2('imputations')[m]), columns = df.columns)
    #     imputed_dfs.append(imputed_df)

    for i in range(m):

        imputed_data = np.array(results.rx2('imputations')[i])
        print(f'Shape of imputed data: {imputed_data.shape}')
        print(f'Shape expected: {(len(df), len(df.columns))}')

        try:

            imputed_df = pd.DataFrame(imputed_data, columns = df.columns, index = df.index)
            imputed_dfs.append(imputed_df)

        except Exception as e:

            print(f'Error in DataFrame construction: {e}')

            if imputed_data.shape [0] == len(df.columns):

                print('Attempting transpose...')
                imputed_df = pd.DataFrame(imputed_data.T, columns = df.columns, index = df.index)
                print('Transposed successfully')
                imputed_dfs.append(imputed_df)

            else:

                raise e

    return imputed_dfs


def evaluate_imputation(imputed_dfs, original_stats):

    original_correlations = original_stats['correlations']
    original_means = original_stats['means']
    orignal_variances = original_stats['variances']

    scores = []
    
    for df in imputed_dfs:

        try:
        
            imp_correlations, imp_means, imp_variances = df[data_cols].corr().values, df.mean(), df.var()

            if(

                np.isnan(imp_correlations).any() or
                np.isnan(imp_means).any() or
                np.isnan(imp_variances).any()

            ): return float('1e10')

            correlations_error = np.mean(np.abs(original_correlations - imp_correlations))
            means_error = np.mean(np.abs(original_means - imp_means))
            variances_error = np.mean(np.abs(orignal_variances - imp_variances))

            imputation_score = np.mean([correlations_error, means_error, variances_error])

            scores.append(imputation_score)
        
        except Exception as e:

            print(f'Error during imputation: {e}')
            return float('1e10')
        
    overall_score = np.mean(scores)
    return overall_score


def objective_function(df, original_stats, m, params):

    try:
        
        imputed_dfs = run_imputation(df, m, params)
        score = evaluate_imputation(imputed_dfs, original_stats)

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

    params = [

        Real(1e-5, 1e-1, name = 'tolerance'),
        Real(n_observations * 0.005, n_observations * 0.01, name = 'empri'),
        Integer(1, 1000, name = 'max_resample')

    ]

    evaluation_scores = []
    
    def wrapped_objective_function(params):

        score = objective_function(df, original_stats, m, params)
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
    
        result.x[0], # tolerance   
        result.x[1], # empri
        int(result.x[2]) # max_resample
    
    ]

    worst_score = max(evaluation_scores)
    final_score = result.fun

    return optimal_parameters, worst_score, final_score