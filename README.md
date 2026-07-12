# Climate-Change-Driven-Compound-Dust-Storm-and-Heatwave-Extremes-in-CA-Under-Future-CMIP6-Scenarios
This repository contains the datasets and Python scripts used in the
study:

  Climate Change-Driven Compound Dust Storm and Heatwave Extremes in
  Central Asia Under Future CMIP6 Scenarios

The repository has been made publicly available to ensure transparency,
reproducibility, and compliance with the PLOS ONE Data and Code
Availability Policy.

Repository Contents

1. Raw Data

This repository includes the datasets used throughout the study.

Historical Period (1992–2021)

-   ERA5 meteorological variables
-   MERRA-2 dust extinction aerosol optical thickness (AOT)
-   Processed datasets used for model development and statistical
    analyses

Future Climate Projections

Bias-corrected CMIP6 outputs for both emission scenarios:

-   SSP2–4.5
-   SSP5–8.5

covering:

-   Near-future (2022–2051)
-   Far-future (2071–2100)

The datasets include the meteorological variables used for heatwave
analysis and future AOT estimation.

2. Python Code

The repository contains all Python scripts developed for this study,
including:

LSTM Model

Python implementation of the Long Short-Term Memory (LSTM) model used to
estimate future atmospheric dust loading (AOT) from meteorological
predictors.

The scripts include:

-   Data preprocessing
-   Model training
-   Hyperparameter tuning
-   Validation
-   Prediction of future AOT
-   Model evaluation

Random Forest Feature Importance

Python scripts used to calculate Random Forest Feature Importance
(RF-FI) for identifying the relative contribution of:

-   Maximum temperature
-   Minimum temperature
-   Precipitation
-   Wind speed

to atmospheric dust extinction AOT.

The scripts reproduce the Feature Importance analyses presented in the
manuscript.

Reproducibility

The datasets and source code provided in this repository are sufficient
to reproduce the principal analyses and figures presented in the
manuscript.

Citation

If you use these data or scripts, please cite:

Broomandi, P., et al. Climate Change-Driven Compound Dust Storm and
Heatwave Extremes in Central Asia Under Future CMIP6 Scenarios. PLOS ONE
(upon publication).
