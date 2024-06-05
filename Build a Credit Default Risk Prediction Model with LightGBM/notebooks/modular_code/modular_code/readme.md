### Introduction

Credit Risk is the possibility of a loss resulting from a borrower's failure to repay a
loan or meet a contractual obligation. The primary goal of a credit risk assessment is to find out whether potential borrowers are creditworthy and have the means to repay their debts so that credit risk or loss can be minimized and the loan is granted to only creditworthy applicants.

If the borrower shows an acceptable level of default risk, then their loan application can be approved upon agreed terms. 

This project involves understanding financial terminologies attached to credit risk and building a classification model for default prediction with LightGBM. Hyperparameter Optimization is done using the Hyperopt library and SHAP is used for model explainability.



#### Folder Structure

input
- credit_risk_data.csv

documents
- project_document.pdf
- lightgbm_explanation.pdf

lib
- model.ipynb
- utils.py
- hyperopt_results.csv

ml_pipeline
- utils.py
- processing.py
- training.py

output

engine.py

requirements.txt

readme.md

#### Steps

- Install dependencies using the command "pip install -r requirements.txt"

- Run engine.py to train and save the model.

