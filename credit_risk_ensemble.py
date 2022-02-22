# Ensemble Learning w/ Loan Stats

## Import Dependencies 
import warnings 
warnings.filterwarnings('ignore') 
import numpy as np 
import pandas as pd 
from pathlib import Path
from collections import Counter 
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion matric 
from imblearn.metrics import classification_report_imbalanced 
from imblearn.ensemble import EasyEnsembleClassifier

## Prepare Data
### Pull csv file path and convert to df 
LoanStats_2019_Q1 = Path('./Resources/LoanStats_2019Q1.csv') 
df = pd.read_csv(LoanStats_2019_Q1)

### Split data into Training and Testing Sets 
X = df.drop(columns='loan_status') 
X = pd.get_dummies(X) 
y = df['loan_status'].to_frame()

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y) 
X_train.shape 

### Scale Training and Testing Data
from sklearn.preproccesing import StandardScaler 
scaler = StandardScaler()
scaler.classification_report_imbalanced(X_train) 
scaler.fit(X_train) 

X_train_scaled = scaler.transform(X_train) 
X_test_scaled = scaler.transform(X_test)
X_train_scaled[:5]

## Balanced Random Forest Classifier 
### Resample the training data with the BalanceRandomForestClassifier 
from imblearn.ensemble import BalancedRandomForestClassifier 
Bal_Ran_For_Class = BalancedRandomForestClassifier(n_estimators=100, random_state=1) 
Bal_Ran_For_Class.fit(X_train, y_train) 

### Use Sklearn to get balanced accuracy score 
y_prediction = Bal_Ran_For_Class(X_teat) 
balanced_accuracy_score(y_test, y_prediction) 

### Use sklearn to display the confusion matrix 
confusion_matrix(y_test, y_pred) 

### Print imbalanced classification report 
print(classification_report_imbalanced(y_test, y_pred)) 

### List feature importance sorted i descending order by feature importance 
feature_id = X.columns
sorted(zip(feature_id, Bal_Ran_For_Class.feature_importances_), reverse = True)

## Classification 
### Training easy ensemble classifier
ESC = EasyEnsembleClassifier(n_estimators = 100,random_state=1) 
ESC = ESC.fit(X_train, y_train) 

### Balanced accuracy score, Confusion Matrix, and Imbalanced classification Report 
y_prediction = ESC.predict(X_test) 
balanced_accuracy_score(y_test, y_prediction) 
confusion_matrix(y_test, y_pred) 
print(classification_report_imbalanced(y_test, y_pred)) 






