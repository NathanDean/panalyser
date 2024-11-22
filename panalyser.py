## Imputation, optimisation and evaluation functions

# Imports

import pandas as pd
import numpy as np

from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical

from linearmodels import PanelOLS, RandomEffects

amelia = importr('Amelia')
pandas2ri.activate()


def validate_inputs(df, data_cols, m, ts, cs, bounds):

    if not isinstance(df, pd.DataFrame):

        raise TypeError('df must be a pandas DataFrame')
    
    if not all(col in df.columns for col in data_cols):

        missing_cols = [col for col in data_cols if col not in df.columns]
        raise ValueError(f'The following columns were not found in the DataFrame: {missing_cols}')
    
    if not isinstance(m, int) or m < 1:

        raise ValueError('m must be a positive integer')
    
    if ts:
        
        if not pd.api.types.is_numeric_dtype(df[ts]):

            raise ValueError('Time series column must be numeric')
        
        if ts not in df.columns:

            raise ValueError(f'Time series column {ts} not found in DataFrame')
    
    if cs and cs not in df.columns:

        raise ValueError(f'Cross-section column {cs} not found in DataFrame')
    
    if bounds is not None and not isinstance(bounds, str):

        raise TypeError('bounds must be an R matrix string')


def clean_data_columns(df, data_cols):

    """Cleans data columns before imputation"""

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


def run_imputation(df, data_cols, m, params, ts = None, cs = None, bounds = None):

    """Creates multiple imputed datasets using Amelia"""

    validate_inputs(df, data_cols, m, ts, cs, bounds)
    df_clean, prob_cols = clean_data_columns(df, data_cols)

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
    
    for col in data_cols:

        if not pd.api.types.is_numeric_dtype(df_clean[col]):

            raise ValueError(f'{col} is not numeric after cleaning')
    
    r_df = pandas2ri.py2rpy(df_clean)

    amelia_params = {

        'x': r_df,
        'm': m,
        'tolerance': params[0],
        'empri': params[1],
        'max_resample': params[2]

    }

    if ts is not None:

        amelia_params['ts'] = ts

    if cs is not None:

        amelia_params['cs'] = cs
    
    if bounds is not None and isinstance(bounds, str):

        amelia_params['bounds'] = robjects.r(bounds)

    results = amelia.amelia(**amelia_params)

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

                print(f'Error during imputation {i + 1}')
                print(f'Error type: {type(e)}')
                print(f'Error message: {str(e)}')

    return imputed_dfs


def evaluate_imputations(imputed_dfs, data_cols, original_stats):

    """Evaluates imputations against original dataset statistics"""

    original_correlations = original_stats['correlations']
    original_means = original_stats['means']
    original_variances = original_stats['variances']

    scores = []
    
    for df in imputed_dfs:

        try:
        
            imp_correlations, imp_means, imp_variances = df[data_cols].corr().values, df[data_cols].mean().values, df[data_cols].var().values

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


def objective_function(df, data_cols, original_stats, m, params, ts = None, cs = None, bounds = None):

    """
    Objective function to be optmised by Bayesian optimiser

    Runs imputation using set of hyperparameters to be tested, then evaluates results and returns score
    """

    try:
        
        imputed_dfs = run_imputation(df, data_cols, m, params, ts, cs, bounds)
        score = evaluate_imputations(imputed_dfs, data_cols, original_stats)

        if np.isinf(score):

            return 1e10

        return score

    except Exception as e:
        
        print(f'Error during optimisation: {e}')
        return 1e10


def optimise_imputation(
        df,
        data_cols,
        m,
        n_calls,
        ts = None,
        cs = None,
        bounds = None,
        tolerance_range = (1e-5, 1e-1),
        empri_factor = (0.005, 0.01),
        max_resample_range = (1, 1000)):

    """
    Tunes Amelia hyperparameters using Bayesian optimiser

    Returns optimal hyperparameter settings as well as worst and best scores

    args:
      df: Pandas dataframe
      data_cols: Columns with missing data
      m: No. of imputations to perform
      n_calls: No. of rounds of optimisation to perform

    """
    
    # No. of observations in dataset
    n_observations = len(df)
    
    # Important statistical properties of original dataset, used as criteria for evaluation
    original_stats = {

        'correlations': df[data_cols].corr().values,
        'means': df[data_cols].mean(),
        'variances': df[data_cols].var()
    
    }

    if (np.isnan(original_stats['correlations']).any() or
        np.isnan(original_stats['means']).any() or
        np.isnan(original_stats['variances']).any()):

            print('Warning: Original correlations, means or variances are NaN')

    # Bounds for Amelia hyperparameters
    params = [

        Real(tolerance_range[0], tolerance_range[1], name = 'tolerance'),
        Real(n_observations * empri_factor[0], n_observations * empri_factor[1], name = 'empri'),
        Integer(max_resample_range[0], max_resample_range[1], name = 'max_resample')

    ]

    evaluation_scores = []
    
    # Returns closure containing params, to be passed to Bayesian optimiser
    def wrapped_objective_function(params):
        
        score = objective_function(df, data_cols, original_stats, m, params, ts, cs, bounds)
        
        if not np.isinf(score):
            evaluation_scores.append(score)
        
        return score

    # Runs Bayesian optimiser w/ params and stores result
    result = gp_minimize(

        wrapped_objective_function,
        params,
        n_calls = n_calls,
        n_random_starts = 5,
        acq_func = 'EI'

    )

    # Stores optimal parameters returned by Bayesian optimiser
    optimal_parameters = [
    
        result.x[0],        # tolerance   
        result.x[1],        # empri
        int(result.x[2])    # max_resample
    
    ]

    worst_score = max(evaluation_scores) if evaluation_scores else float('inf')
    final_score = result.fun

    return optimal_parameters, worst_score, final_score


def prepare_data(df, ts, cs):

    df = df.copy()

    try:

        df[ts] = pd.to_datetime(df[ts])

    except:

        pass

    duplicates = df.duplicated(subset = [cs, ts])

    if duplicates.any():

        raise ValueError(f'Duplicate entries found for {cs}, {ts}')
    
    prepared_df = df.set_index([cs, ts])
    prepared_df = prepared_df.sort_index()

    return prepared_df

def run_fe_regression(df, ts, cs, target, predictors, time_effects = False, cov_type = 'kernel'):

    prepared_df = prepare_data(df, ts, cs)

    model = PanelOLS(

        dependent = prepared_df[target],
        exog = prepared_df[predictors],
        entity_effects = True,   # Uses fixed effects
        time_effects = time_effects

    )

    results = model.fit(cov_type = cov_type)    # Uses Driscoll-Kraay errors

    return results

def run_re_regression(df, ts, cs, target, predictors, cov_type = 'kernel'):

    prepared_df = prepare_data(df, ts, cs)

    model = RandomEffects(

        dependent = prepared_df[target],
        exog = prepared_df[predictors]

    )

    results = model.fit(cov_type = cov_type)

    return results