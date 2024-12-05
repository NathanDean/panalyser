# Panalyser

A Python package for panel data analysis with robust handling of missing data.

## Overview

Panalyser provides tools for analyzing panel data in Python.  It uses the R Amelia provide multiple imputation that accounts for time-series and cross-sectional dependencies, and includes tools for fixed and random effects regression.  It also enables users to combine the results of regression analyses on multiple imputed datasets using Rubin's rules.

Key features:
- Multiple imputation of missing data using Amelia
- Bayesian optimization of Amelia hyperparameters
- Fixed and random effects panel regression
- Combination of regression results using Rubin's rules
- Data validation and cleaning

## Requirements

- Python 3.7+
- R installation
- R packages:
  - Amelia
- Python packages:
  - linearmodels
  - numpy
  - pandas
  - rpy2
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

## Usage

### Basic Imputation

```python
from panalyser import run_imputation

# Run imputation
imputed_dfs = run_imputation(
    df = your_dataframe,
    data_cols = ['column1', 'column2'],
    m = 5,  # number of imputations
    params = [1e-4, 0.01, 100],  # [tolerance, empri, max_resample]
    ts = 'time_series_column',
    cs = 'cross_section_column'
)
```

### Optimized Imputation

```python
from panalyser import optimise_imputation, run_imputation

# Optimize imputation parameters
optimal_params, _, _ = optimise_imputation(
    df = your_dataframe,
    data_cols = ['column1', 'column2'],
    m = 5,
    n_calls = 20,
    ts = 'time_column',
    cs = 'entity_column'
)

# Run imputation with optimal parameters
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

```python
from panalyser import run_combined_regression

# Run fixed effects regression on imputed datasets
results = run_combined_regression(
    imputed_dfs = imputed_dfs,
    ts = 'time_column',
    cs = 'entity_column',
    target = 'dependent_var',
    predictors = ['predictor1', 'predictor2'],
    method = 'fe'  # 'fe' for fixed effects, 're' for random effects
)
```

## How It Works

Panalyser uses Amelia's EMB (Expected Maximization with Bootstrapping) algorithm for multiple imputation. Amelia has been demonstrated to be particularly effective for handling missing data in time-series cross-sectional datasets [1][2]. The algorithm:
- Handles both time-series and cross-sectional dependencies
- Preserves relationships between variables
- Accounts for uncertainty in imputed values

The package then provides tools for panel regression analysis using either fixed or random effects models, which account for the influence of unobserved effects [3][4][5]. These functions use Driscoll-Kraay standard errors by default, as they have been shown to be more robust for panel regressions [6].

Finally, the package allows users to analyse multiple imputed datasets and combine the results using Rubin's rules, accounting for the uncertainty inherent in multiple imputation while providing statistically valid results [7].

## References

[1] J. Honaker and G. King, "What to Do about Missing Values in Time-Series Cross-Sectional Data", American Journal of Political Science, vol. 54, no. 2, pp. 561 – 581, 2010.

[2] J. Honaker, G. King and M. Blackwell, "Amelia II: A Programme for Missing Data", Journal of Statistical Software, vol. 45, no. 7, pp. 1 – 45, 2011.

[3] T.S. Clark and D.A. Linzer, “Should I Use Fixed or Random Effects?”, Political Science Research and Methods, vol. 3, no. 2, pp. 399 – 408, 2015

[4] S. Longhi and A. Nandi, “Analysis of Cross-Section and Panel Data” in A Practical Guide to Using Panel Data, 1st ed. London, UK: Sage Publishing, 2014

[5] Bell and K. Jones, “Explaining Fixed Effects: Random Effects Modeling of Time-Series Cross-Sectional and Panel Data”, Political Science Research and Methods, vol. 3, no. 1, pp. 133 – 153, 2015

[6] D. Hoechle, "Robust standard errors for panel regressions with cross-sectional dependence", The Stata Journal, vol. 7, no. 3, pp. 281 – 312, 2007.

[7] S. van Buuren, "Conclusion" in Flexible Imputation of Missing Data, 2nd ed. New York, NY, USA: Chapman & Hall, 2018.

## License

GPLv2