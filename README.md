Loan Default Prediction Model

A comprehensive credit risk project to predict the probability of default for individual borrowers and calculate the expected loss per loan across a full portfolio. This repository builds, compares, and evaluates three machine learning models, Logistic Regression, Random Forest, and Gradient Boosting, and selects the best performer based on ROC/AUC analysis and 5-fold cross-validation.

The model is applied to a real loan dataset, producing borrower-level default probabilities that feed directly into an expected loss calculator, the same workflow used on a real credit risk desk to price loans and allocate regulatory capital.

Key Features:

Three-Model Comparison: Trains and evaluates Logistic Regression, Random Forest, and Gradient Boosting side by side on the same dataset, with ROC curves for all three overlaid on a single chart for direct comparison.

Cross-Validated AUC Scoring: Each model is validated using 5-fold stratified cross-validation, ensuring performance estimates are reliable and not inflated by a lucky train/test split.

Feature Importance Analysis: Identifies the top predictors of default using logistic regression coefficients. FICO score emerges as the single strongest predictor, by a significant margin, directly motivating the FICO Score Quantization project.

Expected Loss Calculator: For every borrower in the portfolio, the model computes Expected Loss using the standard credit risk formula: EL = PD × LGD × EAD. Results are segmented into low, medium, and high risk profiles to show the full range of credit exposure.

Predicted PD Distribution: Plots the distribution of model-predicted default probabilities split by actual outcome (default vs no-default), showing how cleanly the model separates the two populations.

9-Panel Visualization: A single publication-quality figure covering ROC curves, AUC comparison, confusion matrix, feature importance, PD distribution, expected loss distribution, FICO vs PD scatter, debt-to-income vs PD scatter, and expected loss by risk profile.

Results:

Model Performance:

Model         ,            AUC-ROC 

Logistic Regression    ,    0.835 - Best

Random Forest      ,        0.824

Gradient Boosting    ,      0.818 




Expected Loss by Risk Profile:


Risk Profile     ,         Probability of Default    ,      Expected Loss per Loan

Low-risk borrower      ,         7.2%                 ,        $323                   

Medium-risk borrower    ,        71.5%                ,        $9,655 

High-risk borrower     ,         95.7%                ,       $21,541

Portfolio mean         ,           —                  ,        $13,867

High-risk borrowers carry 67× more expected loss than low-risk borrowers, demonstrating exactly why accurate probability of default estimation is worth investing in.

Tech Stack:

Python 3.x

Pandas: For data loading, cleaning, and feature engineering.

scikit-learn: For Logistic Regression, Random Forest, ROC/AUC evaluation, and 5-fold cross-validation.

XGBoost: For the Gradient Boosting model.

NumPy: For array operations and expected loss calculations.

Matplotlib: For generating the 9-panel analysis chart.

How to Run:

Clone this repository to your local machine.

Install the required libraries: pip install pandas numpy scikit-learn xgboost matplotlib

Ensure Task_3_and_4_Loan_Data.csv is in the same folder as the script.

Run the script: python loan_default_model.py

The script will print model performance metrics to the terminal and save the 9-panel chart as a PNG file.


Future Work:

This project provides a strong foundation for production credit risk modelling. 

Future improvements could include:

Probability Calibration: Applying Platt scaling or isotonic regression to calibrate the raw model probabilities so that a predicted PD of 30% truly corresponds to a 30% observed default rate, critical for regulatory compliance.

Loss Given Default Modelling: Instead of assuming 100% LGD, building a separate regression model to estimate recovery rates by loan type, collateral, and seniority, making the expected loss output more realistic.

Threshold Optimization: Rather than using a default 0.5 classification threshold, optimizing the cutoff using the F1-score or a cost-sensitive metric that accounts for the asymmetric cost of false negatives vs false positives in lending.

Integration with FICO Quantization: Feeding the PD outputs from this model directly into the FICO Score Quantization project to close the loop between individual borrower scoring and portfolio-level rating tier construction.

About:

Built by 

Anurag Kulkarni

Connect on LinkedIn: https://www.linkedin.com/in/anurag-kulkarni97/

GitHub: AnalyticalAnurag97
