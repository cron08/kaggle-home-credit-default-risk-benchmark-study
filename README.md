# Home Credit Default Risk

A machine learning project for predicting repayment difficulty from ~300,000
loan applications in Kaggle's **Home Credit Default Risk** dataset.

I built a LightGBM modelling pipeline that combines application data with
customers' previous loans, credit bureau records, instalment payments, credit
card balances and POS histories. These relational histories are transformed
into customer-level features describing credit exposure, repayment behaviour,
delinquency and financial capacity.

The resulting model achieves ROC-AUC comparable to historical top-5%
leaderboard performance.

The project was subsequently extended beyond the original competition pipeline
to investigate a broader question: **how should complex financial histories be
represented for credit risk prediction?** Published high-performing solutions
provide reference points, while new experiments test statistical feature
engineering, distribution-aware summaries, temporal features and learned
representations under a consistent cross-validation framework.

## Modelling Pipeline

The Home Credit data is distributed across multiple relational tables, with
each applicant associated with variable-length financial histories.

The pipeline:

1. Cleans and validates application and historical data
2. Aggregates one-to-many financial records to the customer level
3. Constructs behavioural and financial features
4. Joins features across application, bureau, previous application, instalment,
   credit card and POS data
5. Trains LightGBM models using stratified K-fold cross-validation
6. Generates out-of-fold predictions for model comparison
7. Evaluates performance using ROC-AUC

Feature engineering includes:

- credit-to-income and annuity-to-income ratios
- credit utilisation and outstanding debt
- payment delays and payment shortfalls
- delinquency frequency and severity
- previous application outcomes
- historical repayment behaviour
- external credit information
- missingness and data-quality indicators

## Extending the Original Approach

Standard solutions to relational tabular problems typically reduce a customer's
history to statistics such as:

`mean · min · max · sum · standard deviation`

While effective, this can discard information about the underlying distribution
and evolution of financial behaviour.

I therefore extended the project to test alternative representations, including:

- **Distribution-aware features** — quantiles, dispersion, skewness, tails and
  threshold exceedances
- **Temporal features** — recency-weighted behaviour, recent-vs-long-term
  differences and trends
- **Statistical feature engineering** — features motivated by the structure and
  distribution of each financial history
- **Representation learning** — learned low-dimensional representations compared
  with hand-engineered features
- **Automated feature engineering** — systematic generation and selection of
  candidate relationships and interactions

Each modification is evaluated through out-of-fold ROC-AUC and ablation
experiments to determine whether it provides a genuine improvement over the
existing pipeline.

## Models and Validation

LightGBM is the primary model because of its performance on large,
heterogeneous tabular datasets and ability to capture nonlinearities and
feature interactions.

Experiments also compare alternative model classes to distinguish improvements
from better feature representation from those caused simply by changing the
predictive model.

Validation includes:

- stratified K-fold cross-validation
- out-of-fold predictions
- train/test distribution-shift analysis
- adversarial validation
- feature importance and model diagnostics
- controlled feature and model ablations

## Results

The LightGBM model achieves ROC-AUC comparable to historical top-5% performance
on the Kaggle Home Credit leaderboard.

Experimental results are reported using a common validation framework so that
the incremental effect of feature and modelling changes can be compared directly.

| Approach | CV ROC-AUC | Kaggle ROC-AUC |
|---|---:|---:|
| Initial LightGBM | — | — |
| Multi-table feature pipeline | — | — |
| Statistical features | — | — |
| Distribution / temporal representations | — | — |
| Final model | — | — |

## Tech

Python · pandas · NumPy · scikit-learn · LightGBM · XGBoost · CatBoost · SHAP

## Data

Data comes from Kaggle's **Home Credit Default Risk** competition. Competition
data is not distributed with this repository.
