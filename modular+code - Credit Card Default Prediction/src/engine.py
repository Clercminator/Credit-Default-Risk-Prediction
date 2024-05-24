import pandas as pd
import numpy as np
import math
from scipy.stats import kurtosis, skew
from scipy import stats, special
import logging
import warnings
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import mean_squared_error, accuracy_score,confusion_matrix, roc_curve, auc, recall_score, precision_score, f1_score,roc_auc_score,auc,roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from ML_pipeline import dataset
#from dataset import *
from ML_pipeline import data_splitting
#from train_test_split import *
from ML_pipeline import data_preprocessing
#from data_preprocessing import *
from ML_pipeline import feature_engineering
#from feature_engineering import *
from ML_pipeline import upsampling_minorityClass
#from upsampling_minorityClass import *
from ML_pipeline import scaling_features
from ML_pipeline import model_params
#from model_params import *
from ML_pipeline import train_model
#from train_model import *
from ML_pipeline import predict_model
#from predict_model import *

print('script started')
# Importing datasets
train, val = dataset.read_data()

# Splitting the train dataset into training and testing
df_test, df_train, y_test, y_train, train, test = data_splitting.training_testing_dataset(train)

# Data preprocessing
train_clean, test_clean, val_clean = data_preprocessing.data_preprocessing(train, test, val, 'NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfTimes90DaysLate', 'MonthlyIncome', 'DebtRatio', 'RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfDependents', 'SeriousDlqin2yrs')

# Feature Engineering 
train_df, test_df, val_df = feature_engineering.feature_engineering(train_clean, test_clean, val_clean, 'NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfTimes90DaysLate', 'NumberOfOpenCreditLinesAndLoans', 'NumberRealEstateLoansOrLines', 'MonthlyIncome', 'NumberOfDependents', 'DebtRatio', 'age')

# Data Transformation
train_x_df, train_y, test_x_df, test_y, val_x_df = upsampling_minorityClass.upsampling_class(train_df, test_df, val_df, False)

# Scaling of the features
train_x_scaled, test_x_scaled, val_x_scaled = scaling_features.scaling_features(train_x_df, test_x_df, val_x_df, False)

# Model Object  
classifier = model_params.model_params()
# Training the model 

model = train_model.train_model(classifier, train_x_scaled, train_y, test_x_scaled, test_y)

# Predicting the model 
val_x_scaled = predict_model.predict_model(model, val_x_scaled)

print('script completed successfully')

path = '../output'
val_x_scaled.to_csv(path+'/'+'test.csv', index=False)
