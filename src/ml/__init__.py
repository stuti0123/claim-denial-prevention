"""
src/ml/__init__.py
------------------
Machine Learning package for the Claim Denial System — Week 5.

Modules
-------
trainer   : Trains, evaluates, and saves the XGBoost denial prediction model.
predictor : Loads the saved model and runs inference on new claims.
explainer : SHAP-based feature attribution — explains WHY a claim is risky.

Why XGBoost?
------------
1. Handles missing values natively (no imputation needed for most features).
2. Fast inference — suitable for real-time API calls.
3. Built-in feature importance — feeds directly into the SHAP explainer.
4. Proven on tabular healthcare data; outperforms Logistic Regression on
   non-linear patterns without the cost of deep learning.
5. Low memory footprint compared to neural networks.
"""
