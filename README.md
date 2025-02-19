# Panalyser

A Python package for user-friendly panel data analysis with robust handling of missing data.

## Overview

Panalyser provides user-friendly tools for performing panel data analysis in Python.  It uses the R package Amelia to perform multiple imputation which accounts for time-series and cross-sectional dependencies, and can tune Amelia's hyperparameters using Bayesian optimisation.  It also includes tools for fixed and random effects linear regression, and allows users to combine the results of regression analyses on multiple imputed datasets using Rubin's rules.

The package currently allows users to set the following Amelia hyperparameters:
- x
- m
- ts
- cs
- bounds
- tolerance
- empri
- max.resample

<!-- Key features:
- Multiple imputation of missing data using Amelia
- Bayesian optimization of Amelia hyperparameters
- Fixed and random effects panel regression
- Combination of regression results using Rubin's rules
- Data validation and cleaning -->

## Requirements

- Python 3.7+
- R installation
- R packages:
  - Amelia
- Python packages:
  - numpy
  - pandas
  - rpy2
  - linearmodels
  - scikit-optimize
  - scipy

## Installation

```bash
pip install panalyser
```

Panalyser requires R and Amelia to be installed. You can install Amelia in R using:
```r
install.packages("Amelia")
```

You can find instructions on installing R [here](https://www.r-project.org/), and the Amelia II documentation [here](https://www.rdocumentation.org/packages/Amelia/versions/1.8.1).

## Usage

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

## How It Works

Panalyser uses rpy2 to access Amelia in R.  This gives Python users access to robust multiple imputation for panel data which:
- handles time-series and cross-sectional dependencies
- preserves relationships across variables, and
- accounts for uncertainty in imputed data [1][2].

Amelia imputation is supported by comprehensive data cleaning and formatting functions to minimise the potential for errors in the translation between Python and R.

The optimisation function uses the gp_minimize function from scikit-optimize. It aims to minimise the difference between the correlations, means, and variances of the original and imputed datasets, to find the set of hyperparameters that best preserves the statistical properties of the observed data.

The package uses linearmodels to provide linear regression with fixed and random effects, allowing users to account for the influence of unobserved effects in their data [3][4][5]. Panalyser uses Driscoll-Kraay standard errors by default, which have been shown to be more robust for linear regression on panel data [6].

Finally, the package allows users to analyse multiple imputed datasets and combine the results using Rubin's rules.  This enables users to account for the uncertainty created by imputing data while providing statistically valid results [7].

## Why this approach?

This package began as part of a project that involved analysing a sparse panel dataset.

While some advanced machine learning techniques can handle missing data, they provide results which are hard to interpret, and have limited use in establishing clear relationships between variables [8].

As I began looking into options for handling the missing data, I learned that standard multiple imputation (MI) or maximum likelihood approaches are not valid on panel data [9].  The only tool I found that could handle missing panel data in a robust way was Amelia II.

As my project required a solution in Python, I built a bridge from Python to R using rpy2.  This was a difficult process, involving a lot of error handling and bug fixing, and it occurred to me that others might find it useful to have a package which handled this setup for them.

When I came to analyse my imputed data, I found that although tools for linear regression on panel data existed in linearmodels, they were not set up to use the most rigorous parameter settings.  I also found the relevant functions to be obscurely named and difficult to understand without intensively studying the package's documentation.  I therefore decided to repackage these functions in Panalyser, setting them up to use Driscoll-Kraay standard errors by default, and giving the functions and their attributes more meaningful names.

Similarly, I found tools for combining multiple imputed datasets to be obscure and difficult to use, so decided to include a straightforward function for achieving this in the package as well.

Overall, I aimed to create a package which brings together tools for panel data analysis in a way that is user-friendly and statistically robust, and ultimately makes it easier for researchers to gain valuable insights from panel data.

## Next steps

Panalyser currently only allows users to tune a small number of Amelia's hyperparameters.  Future work on the package will focus on allowing users to set all the possible hyperparameters, and tune them using the optimisation function.

## References

[1] J. Honaker and G. King, "What to Do about Missing Values in Time-Series Cross-Sectional Data", American Journal of Political Science, vol. 54, no. 2, pp. 561 – 581, 2010.

[2] J. Honaker, G. King and M. Blackwell, "Amelia II: A Programme for Missing Data", Journal of Statistical Software, vol. 45, no. 7, pp. 1 – 45, 2011.

[3] T.S. Clark and D.A. Linzer, “Should I Use Fixed or Random Effects?”, Political Science Research and Methods, vol. 3, no. 2, pp. 399 – 408, 2015

[4] S. Longhi and A. Nandi, “Analysis of Cross-Section and Panel Data” in A Practical Guide to Using Panel Data, 1st ed. London, UK: Sage Publishing, 2014

[5] Bell and K. Jones, “Explaining Fixed Effects: Random Effects Modeling of Time-Series Cross-Sectional and Panel Data”, Political Science Research and Methods, vol. 3, no. 1, pp. 133 – 153, 2015

[6] D. Hoechle, "Robust standard errors for panel regressions with cross-sectional dependence", The Stata Journal, vol. 7, no. 3, pp. 281 – 312, 2007.

[7] S. van Buuren, "Conclusion" in Flexible Imputation of Missing Data, 2nd ed. New York, NY, USA: Chapman & Hall, 2018.

[8] M. Bertolini, D. Mezzogori, M. Neroni, F. Zammori, “Machine Learning for industrial purposes”, Expert Systems with Applications, vol. 175, 2021

[9] R.J.A. Little and D.B. Rubin, “Multivariate Normal Examples, Ignoring the Missingness Mechanism” in Statistical Analysis with Missing Data, 3rd ed. Hoboken, NJ, USA: Wiley, 2020

## License

GPLv2