"""

Panel Data Analysis Helper Module

This module provides tools for handling panel data, including multiple imputation, hyperparameter optimization, and panel regression analysis. 

It uses rpy2 to access the R package Amelia for imputation, and uses Python's linearmodels for panel regression.

Main Features:
- Multiple imputation of missing data using Amelia
- Bayesian optimization of Amelia hyperparameters
- Fixed and random effects panel regression
- Data validation and cleaning utilities

"""


# Required imports

import pandas as pd
import numpy as np

from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

from skopt import gp_minimize
from skopt.space import Real, Integer

from linearmodels import PanelOLS, RandomEffects


# Initialise R interface

amelia = importr('Amelia')
pandas2ri.activate()


def validate_inputs(df, data_cols, m, ts, cs, bounds):

    """

    Validates input parameters for imputation functions.
    
    Args:
        df (pandas.DataFrame): Input dataset
        data_cols (list): List of column names containing data to be imputed
        m (int): Number of imputations to perform
        ts (str): Time series column
        cs (str): Cross-section column
        bounds (str): R matrix string specifying bounds for imputation
    
    Raises:
        TypeError: If inputs are of incorrect type
        ValueError: If inputs contain invalid values or missing columns
    
    """

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

    """

    Cleans data columns before imputation by converting to numeric values.
    
    Args:
        df (pandas.DataFrame): Input dataset
        data_cols (list): List of column names to clean
    
    Returns:
        tuple: (cleaned DataFrame, list of problematic columns with details)
    
    """

    df_clean = df.copy()
    prob_cols = []

    for col in data_cols:

        try:

            # Remove non-numeric characters and convert to numeric
            df_clean[col] = df_clean[col].astype(str).str.strip()
            df_clean[col] = df_clean[col].replace(r'[^0-9.-]', '', regex = True)
            numeric_col = pd.to_numeric(df_clean[col], errors = 'coerce')
            df_clean[col] = numeric_col

            # Check for high percentage of null values
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

    """
    
    Performs multiple imputation using Amelia.
    
    Args:
        df (pandas.DataFrame): Input dataset
        data_cols (list): Columns containing data to be imputed
        m (int): Number of imputations to perform
        params (list): Amelia hyperparameters [tolerance, empri, max_resample]
        ts (str, optional): Time series column
        cs (str, optional): Cross-section column
        bounds (str, optional): R matrix string specifying bounds
    
    Returns:
        list: List of imputed DataFrames
    
    Raises:
        ValueError: If data cleaning or imputation fails
    
    """

    validate_inputs(df, data_cols, m, ts, cs, bounds)
    df_clean, prob_cols = clean_data_columns(df, data_cols)

    # Handle problematic columns
    if prob_cols:

        print('Warning: Problems in following columns:')

        for col in prob_cols:

            print(f'Column: {col}')
            print(f'Sample values: {col["sample"]}')

            if 'percentage_null' in col:

                print(f'Percentage null: {col['percentage_null']}')

            if 'error' in col:

                print(f'Error: {col['error']}')

        print('Continuing imputation with cleaned data')
    
    # Verify data columns are numeric after cleaning
    for col in data_cols:

        if not pd.api.types.is_numeric_dtype(df_clean[col]):

            raise ValueError(f'{col} is not numeric after cleaning')
    
    # Convert pandas DataFrame to R DataFrame
    r_df = pandas2ri.py2rpy(df_clean)

    # Set up Amelia hyperparameters
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

    # Run imputation
    results = amelia.amelia(**amelia_params)

    imputed_dfs = []

    # Process results
    for i in range(m):

        try:

            imputed_data = np.array(results.rx2('imputations')[i])
            imputed_data_shape = imputed_data.shape
            expected_data_shape = (len(df), len(df.columns))

            # Handle potential data shape issues caused by translation between R and Python
            if imputed_data_shape != expected_data_shape:

                if imputed_data_shape[0] == expected_data_shape[1] and imputed_data_shape[1] == expected_data_shape[0]:

                    imputed_data = imputed_data.T

                else:

                    raise ValueError(f'Error: Expected data shape {expected_data_shape}, got {imputed_data_shape}')
                
            imputed_df = pd.DataFrame(imputed_data, columns = df.columns, index = df.index)

            # Verify data columns are numeric after imputation
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

    """
    
    Evaluates imputation quality by comparing statistical properties of imputed and original datasetes.
    
    Args:
        imputed_dfs (list): List of imputed DataFrames
        data_cols (list): Columns to evaluate
        original_stats (dict): Statistical properties of original dataset
    
    Returns:
        float: Average deviation of statistical properties of imputed and original datasets (lower is better)
    
    """

    original_correlations = original_stats['correlations']
    original_means = original_stats['means']
    original_variances = original_stats['variances']

    scores = []
    
    for df in imputed_dfs:

        try:
        
            imp_correlations, imp_means, imp_variances = df[data_cols].corr().values, df[data_cols].mean().values, df[data_cols].var().values

            # Check for NaN values in statistical property variables
            if(

                np.isnan(imp_correlations).any() or
                np.isnan(imp_means).any() or
                np.isnan(imp_variances).any()

            ):
                
                scores.append(float('inf'))
                continue

            # Calculate errors using mean absolute distance
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
    
    Objective function for Bayesian optimization of Amelia hyperparameters.
    
    Args:
        df (pandas.DataFrame): Input dataset
        data_cols (list): Columns to impute
        original_stats (dict): Statistical properties of original dataset
        m (int): Number of imputations Number of imputations to perform
        params (list): Amelia hyperparameters [tolerance, empri, max_resample]
        ts (str, optional): Time series column
        cs (str, optional): Cross-section column
        bounds (str, optional): R matrix string specifying bounds
    
    Returns:
        float: Average deviation of statistical properties of imputed and original datasets (lower is better)
    
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

    Tunes Amelia hyperparameters using Bayesian optimization.
    
    Args:
        df (pandas.DataFrame): Input dataset
        data_cols (list): Columns to impute
        m (int): Number of imputations
        n_calls (int): Number of optimization iterations
        ts (str, optional): Time series column
        cs (str, optional): Cross-section column
        bounds (str, optional): R matrix string specifying bounds
        tolerance_range (tuple): Range for convergence tolerance
        empri_factor (tuple): Range for empirical prior factor
        max_resample_range (tuple): Range for maximum resampling attempts
    
    Returns:
        tuple: (optimal parameters, worst score, final score)

    """
    
    # No. of observations in dataset
    n_observations = len(df)
    
    # Calculate statistical properties of original dataset
    original_stats = {

        'correlations': df[data_cols].corr().values,
        'means': df[data_cols].mean(),
        'variances': df[data_cols].var()
    
    }

    if (np.isnan(original_stats['correlations']).any() or
        np.isnan(original_stats['means']).any() or
        np.isnan(original_stats['variances']).any()):

            print('Warning: Original correlations, means or variances are NaN')

    # Define hyperparameter space for optimisation
    params = [

        Real(tolerance_range[0], tolerance_range[1], name = 'tolerance'),
        Real(n_observations * empri_factor[0], n_observations * empri_factor[1], name = 'empri'),
        Integer(max_resample_range[0], max_resample_range[1], name = 'max_resample')

    ]

    evaluation_scores = []
    
    def wrapped_objective_function(params):
        
        score = objective_function(df, data_cols, original_stats, m, params, ts, cs, bounds)
        
        if not np.isinf(score):
            evaluation_scores.append(score)
        
        return score

    # Run Bayesian optimisation
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

    """

    Prepares panel data for regression analysis.
    
    Args:
        df (pandas.DataFrame): Input dataset
        ts (str): Time series column
        cs (str): Cross-section column
    
    Returns:
        pandas.DataFrame: Prepared dataset with multi-index
    
    Raises:
        ValueError: If duplicate entries are found

    """

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

    """
    
    Runs fixed effects panel regression.
    
    Args:
        df (pandas.DataFrame): Input dataset
        ts (str): Time series column
        cs (str): Cross-section column
        target (list): Target variable(s)
        predictors (list): Predictor variable(s)
        time_effects (bool, optional): Whether to include time fixed effects
        cov_type (str, optional): Covariance estimator type, default 'kernel' for Driscoll-Kraay errors
    
    Returns:
        linearmodels.PanelResults: Regression results
    
    """
        

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

    """
    
    Runs random effects panel regression.
    
    Args:
        df (pandas.DataFrame): Input dataset
        ts (str): Time series column
        cs (str): Cross-section column
        target (list): Target variable(s)
        predictors (list): Predictor variable(s)
        cov_type (str, optional): Covariance estimator type, default 'kernel' for Driscoll-Kraay errors
    
    Returns:
        linearmodels.PanelResults: Regression results
    
    """

    prepared_df = prepare_data(df, ts, cs)

    model = RandomEffects(

        dependent = prepared_df[target],
        exog = prepared_df[predictors]

    )

    results = model.fit(cov_type = cov_type)

    return results