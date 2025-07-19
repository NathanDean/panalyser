# Panalyser

## Overview

A package to make statistically valid imputation and analysis of panel data easier in Python.

### What it does

- Cleans and validates data
- Handles missing data via multiple imputation using R package Amelia II (via Python-R bridge)
- Tunes Amelia II's hyperparameters using Bayesian optimisation
- Estimates target values using fixed or random effects linear regression
- Combines regression results using Rubin's rules

### Motivation

Panel data analysis is complicated, as standard statistical tools cannot handle the time-series and cross-sectional dependencies it introduces. Therefore we need specialist tools, such as Amelia II for multiple imputation, and fixed or random effects regression models, such as those provided by linearmodels.

However, while these tools exist, they can be very difficult to use. Accessing Amelia II from Python requires setting up a Python-R bridge, which is fraught with potential errors. Selecting the right linear model and applying the correct hyperparameters requires extensive reading of documentation and advanced statistical knowledge.

Panalyser handles the Python-R setup and data validation to minimise the potential for errors. It also provides easily understandable wrappers for optimisation and analysis functions, with semantically meaningful names and correct hyperparameters as defaults.

In doing so, this package aims to make statistically robust imputation and analysis of panel data easier and more accessible to non-experts.

### Tech stack

- Python
- rpy2 (for Python-R bridge)
- Amelia II (for multiple imputation)
- scikit-optimize (for Bayesian optimisation)
- linearmodels (for fixed and random effects linear regression)

## Installation

To use Panalyser, download this repo and install the following:

- Python 3.13+
- R installation
- R packages:
  - Amelia II
- Python packages:
  - numpy
  - pandas
  - rpy2
  - linearmodels
  - scikit-optimize
  - scipy

You can find instructions on installing R [here](https://www.r-project.org/), and the Amelia II documentation [here](https://www.rdocumentation.org/packages/Amelia/versions/1.8.1).

## Usage

Follow the instructions below to perform imputation, optimisation and analysis using Panalyser.

### Basic Imputation

Performs multiple imputation on a DataFrame, and returns a list of complete DataFrames with missing data imputed.

```python
from panalyser import run_imputation

# Runs imputation, returns list of imputed DataFrames
imputed_dfs = run_imputation(
    df = your_dataframe, # DataFrame with missing data
    data_cols = ['column1', 'column2'], # Array of the columns with missing data
    m = 5,  # number of imputations
    params = [1e-4, 0.01, 100],  # Amelia hyperparameters [tolerance, empri, max_resample]
    ts = 'time_series_column', # Time-series column, optional, default None
    cs = 'cross_section_column', # Cross-section column, optional, default None
    bounds = None # R matrix string specifying bounds, optional, default None
)

```

N.B. Panalyser currently allows users to set the following Amelia II hyperparameters:

- x
- m
- ts
- cs
- bounds
- tolerance
- empri
- max.resample

Find more info on how to use Amelia II in the docs [here](https://www.rdocumentation.org/packages/Amelia/versions/1.8.1).

### Optimized Imputation

Uses Bayesian optimisation algorithm to find hyperparameters which best preserve statistical properties of DataFrame, then uses them to perform multiple imputation.

```python
from panalyser import optimise_imputation, run_imputation

# Optimises imputation parameters, returns tuple (optimal parameters, worst score, final score)
optimal_params, _, _ = optimise_imputation(
    df = your_dataframe,
    data_cols = ['column1', 'column2'],
    m = 5,
    n_calls = 20, # No. of iterations for optimisation algorithm to perform
    ts = 'time_column',
    cs = 'entity_column',
    bounds = None, # Optional, default value None
    tolerance_range = (1e-5, 1e-1), # Optional, default value (1e-5, 1e-1)
    empri_factor = (0.005, 0.01), # Optional, default value (0.005, 0.01)
    max_resample_range = (1, 1000)): # Optional, default value (1, 1000)
)

# Runs imputation with optimal parameters
imputed_dfs = run_imputation(
    df = your_dataframe,
    data_cols = ['column1', 'column2'],
    m = 5,
    params = optimal_params,
    ts = 'time_column',
    cs = 'entity_column'
)
```

### Panel Regression

Takes a list of DataFrames with imputed data, analyses them using linear regression with fixed or random effects, and combines the results using Rubin's rules.

```python
from panalyser import run_combined_regression

# Runs fixed effects regression on imputed datasets, returns dictionary of combined regression results
results = run_combined_regression(
    imputed_dfs = imputed_dfs, # List of imputed DataFrames
    ts = 'time_column', # Time-series column
    cs = 'entity_column', # Cross-section column
    target = ['dependent_var'], # Dependent variable
    predictors = ['predictor_1', 'predictor_2'], # Independent variables
    method = 'fe',  # 'fe' for fixed effects, 're' for random effects
    time_effects = False, # Time fixed effects (only relevant if running fixed effects), optional, default False
    cov_type = 'kernel' # Covariance estimator type, optional, default 'kernel' for Driscoll-Kraay errors
)
```

You can also run linear regression with fixed or random effects on a single DataFrame.

```Python
from panalyser import run_fe_regression, run_re_regression

# Runs fixed fixed effects regression, returns linearmodels PanelResults
results = run_fe_regression(
    df = df,
    ts = 'time_series_column'
    cs = 'cross_section_column',
    target = ['dependent_var'],
    predictors = ['predictor_1', 'predictor_2'],
    time_effects = False, # Optional, default False
    cov_type = 'kernel' # Optional, default 'kernel'
)

# Runs random effects regression, returns linearmodels PanelResults
results = run_re_regression(
    df = df,
    ts = 'time_series_column'
    cs = 'cross_section_column',
    target = ['dependent_var'],
    predictors = ['predictor_1', 'predictor_2'],
    cov_type = 'kernel' # Optional, default 'kernel'
)

```

## Next steps

Panalyser evolved from code I created for an MSc project requiring panel data analysis. The code has been genericised to allow it to work with other datasets, and it can be used for valid imputation and analysis by following the instructions below. However it needs further work before being officially released as a package:

- **Python verson compatibility:** Panalyser has only been tested using Python 3.13. This is overly restrictive, and while it believe it should work with more commonly used earlier releases, this needs to be tested before a wider release.
- **Modularisation:** Currently all functions are contained in the same file. This isn't ideal from the perspective of maintainability, so the different parts of the package should be split into their own modules.
- **Testing:** I have tested Panalyser with several different datasets and it appears to perform as expected. However, more extensive automated testing is needed to confirm it behaves correctly across scenarios including edge cases before it can be considered fully reliable.
- **Error handling:** Error messages are currently quite generic, and should be made more meaningful before wider release to help users debug any problems that arise.
- **Additional features:** Panalyser is already a valuable tool for many use cases. However additional features such as more robust data validation for edge cases, allowing tuning of all Amelia II hyperparameters, and enabling parallel processing for large datasets, would make this package more useful and user-friendly for real-world problems.

My time to implement these improvements is currently limited while I work on other projects for my MSc. However I intend to revisit Panalyser in future to address these issues.

## License

GPLv2
