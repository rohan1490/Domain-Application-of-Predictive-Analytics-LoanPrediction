# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 00:31:07 2021

@author: rohan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


scaler = MinMaxScaler()

df=pd.read_csv('D:/Study/SEM 2/Domain Application of Predictive Analytics/Project/Dataset/train.csv')
df=df.drop('Loan_ID',1)
df.describe()

#Engineering Gender
df.loc[df['Gender'] == 'Male', 'Gender'] = 1
df.loc[df['Gender'] == 'Female', 'Gender'] = 2
df['Gender'] = df['Gender'].replace(np.nan, 1) 
df["Gender"] = pd.to_numeric(df["Gender"])
df['Gender'].value_counts()

#Engineering Married
df.loc[df['Married'] == 'Yes', 'Married'] = 1
df.loc[df['Married'] == 'No', 'Married'] = 2
df['Married'] = df['Married'].replace(np.nan, 1) 
df["Married"] = pd.to_numeric(df["Married"])
df['Married'].value_counts()


#Engineering Dependents
df.loc[df['Dependents'] == '3+', 'Dependents'] = 3
df['Dependents'] = df['Dependents'].replace(np.nan, 0) 
df["Dependents"] = pd.to_numeric(df["Dependents"])
df['Dependents'].value_counts()


#Engineering Education
df.loc[df['Education'] == 'Graduate', 'Education'] = 1
df.loc[df['Education'] == 'Not Graduate', 'Education'] = 2
df["Education"] = pd.to_numeric(df["Education"])
df['Education'].value_counts()


#Engineering Selfemployed
df.loc[df['Self_Employed'] == 'Yes', 'Self_Employed'] = 1
df.loc[df['Self_Employed'] == 'No', 'Self_Employed'] = 2
df['Self_Employed'] = df['Self_Employed'].replace(np.nan, 3) 
df["Self_Employed"] = pd.to_numeric(df["Self_Employed"])
df['Self_Employed'].value_counts()


#Engineering LoanAmount
print(df['LoanAmount'].median())
#df['LoanAmount'] = df['LoanAmount'].replace(np.nan, 1) 
df['LoanAmount'] = df['LoanAmount'].replace(np.nan, 128) 
df["LoanAmount"] = pd.to_numeric(df["LoanAmount"])
df['LoanAmount'].isna().value_counts()


#Engineering Load_Amount_Term
#Categories
#Btw 0 to 120 months = 1
#Btw 121 to 180 months = 2
#Btw 181 to 300 months = 3
#Btw 301 to 360 months = 4
#Btw 361 to 480 months = 5

df['Loan_Amount_Term'].value_counts()
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].replace(np.nan, 1) 
df.loc[df['Loan_Amount_Term'] < 121, 'Loan_Amount_Term'] = 1
df.loc[((df['Loan_Amount_Term'] > 120)&(df['Loan_Amount_Term'] < 181)), 'Loan_Amount_Term'] = 2
df.loc[((df['Loan_Amount_Term'] > 180)&(df['Loan_Amount_Term'] < 301)), 'Loan_Amount_Term'] = 3
df.loc[((df['Loan_Amount_Term'] > 300)&(df['Loan_Amount_Term'] < 361)), 'Loan_Amount_Term'] = 4
df.loc[((df['Loan_Amount_Term'] > 360)&(df['Loan_Amount_Term'] < 481)), 'Loan_Amount_Term'] = 5
df["Loan_Amount_Term"] = pd.to_numeric(df["Loan_Amount_Term"])
df['Loan_Amount_Term'].value_counts()


#Engineering Credit_History
df['Credit_History'].value_counts()
df['Credit_History'] = df['Credit_History'].replace(np.nan, 2) 
df['Credit_History'].value_counts()
df["Credit_History"] = pd.to_numeric(df["Credit_History"])


#Engineering Property_Area
df['Property_Area'].value_counts()
df.loc[df['Property_Area'] == 'Urban', 'Property_Area'] = 1
df.loc[df['Property_Area'] == 'Semiurban', 'Property_Area'] = 2
df.loc[df['Property_Area'] == 'Rural', 'Property_Area'] = 3
df['Property_Area'].value_counts()
df["Property_Area"] = pd.to_numeric(df["Property_Area"])



#Engineering Loan_Status
df['Loan_Status'].value_counts()
df.loc[df['Loan_Status'] == 'Y', 'Loan_Status'] = 1
df.loc[df['Loan_Status'] == 'N', 'Loan_Status'] = 0
df['Loan_Status'].value_counts()
df["Loan_Status"] = pd.to_numeric(df["Loan_Status"])


print(round(df.corr(),2))
sns.heatmap(df.corr())


df.describe()
df.dtypes
df['Gender'] = df['Gender'].astype('category')
df['Married'] = df['Married'].astype('category')
df['Dependents'] = df['Dependents'].astype('category')
df['Education'] = df['Education'].astype('category')
df['Self_Employed'] = df['Self_Employed'].astype('category')
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].astype('category')
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].astype('category')


#MinMax Normalization
df[['ApplicantIncome', 'CoapplicantIncome','LoanAmount']] = scaler.fit_transform(df[['ApplicantIncome', 'CoapplicantIncome','LoanAmount']])



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('Loan_Status',axis=1), df['Loan_Status'], test_size=0.35, random_state=101, stratify=df['Loan_Status'])
                                    
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)

import sklearn.metrics as metrics
probs = logmodel.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
print(classification_report(y_test,predictions))
print(accuracy_score(y_test,predictions))
confusion_matrix = pd.crosstab(y_test,predictions)
sns.heatmap(confusion_matrix, annot=True, fmt = ".0f",cmap='Wistia', linewidths=.5)


# tree = DecisionTreeClassifier()
# tree.fit(X_train, y_train)
# ypred_tree = tree.predict(X_test)
# print(classification_report(y_test,ypred_tree))
# print(accuracy_score(y_test,ypred_tree))
# pd.crosstab(y_test,ypred_tree)


# forest=RandomForestClassifier()
# forest.fit(X_train, y_train)
# ypred_forest = forest.predict(X_test)
# print(classification_report(y_test,ypred_forest))
# print(accuracy_score(y_test,ypred_forest))
# pd.crosstab(y_test,ypred_forest)

from sklearn.svm import SVC
# classifier = SVC(kernel='rbf', random_state = 1)
# classifier.fit(X_train,y_train)
# ypred_svm = classifier.predict(X_test)
# print(classification_report(y_test,ypred_svm))
# print(accuracy_score(y_test,ypred_svm))
# pd.crosstab(y_test,ypred_svm)


#Using SMOTE
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=12)
X_train_sm, y_train_sm = sm.fit_resample(X_train.astype('float64'), y_train)

logmodel = LogisticRegression()
logmodel.fit(X_train_sm,y_train_sm)
predictions = logmodel.predict(X_test)
print(classification_report(y_test,predictions))
print(accuracy_score(y_test,predictions))

# tree = DecisionTreeClassifier()
# tree.fit(X_train_sm, y_train_sm)
# ypred_tree = tree.predict(X_test)
# print(classification_report(y_test,ypred_tree))
# print(accuracy_score(y_test,ypred_tree))
# pd.crosstab(y_test,ypred_tree)

# forest=RandomForestClassifier()
# forest.fit(X_train_sm, y_train_sm)
# ypred_forest = forest.predict(X_test)
# print(classification_report(y_test,ypred_forest))
# print(accuracy_score(y_test,ypred_forest))
# pd.crosstab(y_test,ypred_forest)


# classifier = SVC(kernel='rbf', random_state = 1)
# classifier.fit(X_train_sm,y_train_sm)
# ypred_svm = classifier.predict(X_test)
# print(classification_report(y_test,ypred_svm))
# print(accuracy_score(y_test,ypred_svm))


#Using SMOTEENN
from imblearn.combine import SMOTEENN
sm = SMOTEENN(random_state=12)
X_train_sm, y_train_sm = sm.fit_resample(X_train.astype('float64'), y_train)

logmodel = LogisticRegression()
logmodel.fit(X_train_sm,y_train_sm)
predictions = logmodel.predict(X_test)
print(classification_report(y_test,predictions))
print(accuracy_score(y_test,predictions))


tree = DecisionTreeClassifier()
tree.fit(X_train_sm, y_train_sm)
ypred_tree = tree.predict(X_test)
print(classification_report(y_test,ypred_tree))
print(accuracy_score(y_test,ypred_tree))
pd.crosstab(y_test,ypred_tree)

forest=RandomForestClassifier()
forest.fit(X_train_sm, y_train_sm)
ypred_forest = forest.predict(X_test)
print(classification_report(y_test,ypred_forest))
print(accuracy_score(y_test,ypred_forest))
pd.crosstab(y_test,ypred_forest)


classifier = SVC(kernel='rbf', random_state = 1)
classifier.fit(X_train_sm,y_train_sm)
ypred_svm = classifier.predict(X_test)
print(classification_report(y_test,ypred_svm))
print(accuracy_score(y_test,ypred_svm))
pd.crosstab(y_test,ypred_svm)



#LightGBM using initial dataset
from lightgbm import LGBMClassifier
from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

model= LGBMClassifier()

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

# fit the model on the whole dataset
model = LGBMClassifier()
model.fit(X_train, y_train)
# make prediction
lgm_pred = model.predict(X_test)
print(classification_report(y_test,lgm_pred))
print(accuracy_score(y_test,lgm_pred))
pd.crosstab(y_test,lgm_pred)

