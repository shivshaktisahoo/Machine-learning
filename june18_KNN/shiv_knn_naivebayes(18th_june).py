# -*- coding: utf-8 -*-
"""Shiv_KNN_NaiveBayes(18th june).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lvF2_AYf2GlrOuFlpFy1yZ2zXF65KCL5

#### Datasets :
1. Indian Liver Patient Records  - https://www.kaggle.com/uciml/indian-liver-patient-records

2. Productivity Prediction of Garment Employees Data Set - https://archive.ics.uci.edu/ml/datasets/Productivity+Prediction+of+Garment+Employees
"""

import numpy as np
import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

"""#### **1. Indian Liver Patient Records** 
  - Prediction with KNN Algorithm
  - Prediction with Naive Bayes Algorithm
"""

ilp_dataset = pd.read_csv('https://raw.githubusercontent.com/shivshaktisahoo/test/main/indian_liver_patient.csv')
ilp_dataset.head()

ilp_dataset.info()
ilp_dataset.Albumin_and_Globulin_Ratio.describe()
ilp_dataset['Albumin_and_Globulin_Ratio'][240:256]
ilp_dataset['Albumin_and_Globulin_Ratio'] = ilp_dataset['Albumin_and_Globulin_Ratio'].fillna(round(ilp_dataset.Albumin_and_Globulin_Ratio.describe()['mean'], 2))
ilp_dataset['Gender'] = ilp_dataset['Gender'].map({"Male":1, "Female":0})
ilp_dataset = ilp_dataset.rename(columns={'Dataset':'LiverPatient'})
ilp_dataset['LiverPatient'] = ilp_dataset['LiverPatient'].map({1:1, 2:0})

ilp_dataset

ilp_corr = ilp_dataset.corr()
ilp_corr

# relationship between attribute and target with minimum threshold
MIN_THRES = 0.04         
final_attr = []
for i in ilp_corr.drop(['LiverPatient'],axis=1):
  if abs(ilp_corr[i]['LiverPatient']) >= MIN_THRES:
      final_attr.append(i)
final_attr

ilp_corr = ilp_dataset[final_attr].corr()
ilp_corr

ilp_corr["Age"]["Gender"]

# relationship between attribute and attribute with maximum threshold
MAX_THRES = 0.85
list1 = list(ilp_corr.columns)
max_final_attr = []
for i in enumerate(list1):
  for j in list1[i[0]+1:]:
    if abs(ilp_corr[i[1]][j]) >= MAX_THRES:
      print(abs(ilp_corr[i[1]][j]),i[1],j)
      max_final_attr.append(i[1])
max_final_attr

X1 = ilp_dataset.drop(['LiverPatient','Total_Bilirubin'], axis=1)
Y1 = ilp_dataset['LiverPatient']
neigh = KNeighborsClassifier(n_neighbors=20)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X1,Y1,test_size=0.1)

neigh.fit(x_train,y_train)
neigh.score(x_test,y_test)

print(neigh.predict_proba(np.array([65,0,0.1,187,16,18,6.8,3.3,0.90]).reshape(1,-1)))
print(neigh.predict_proba(np.array([34,1,0.6,67,21,9,8.8,6.3,0.6]).reshape(1,-1)))

"""- Naive bayes Algorithm

"""

from sklearn.naive_bayes import GaussianNB
neigh = GaussianNB()
neigh.fit(x_train, y_train)

prediction = neigh.predict_proba(np.array([65,0,0.1,187,16,18,6.8,3.3,0.90]).reshape(1,-1))[0]
predicted_target = np.argmax(prediction)
predicted_percentage = round(np.max(prediction)*100,2)
print(f"Their is {predicted_percentage}% chance of target {predicted_target}")

"""#### **2. Productivity Prediction of Garment Employees Data Set**
- Prediction with KNN Algorithm
- Prediction with Naive Bayes Algorithm
"""

garments_dataset = pd.read_csv('https://raw.githubusercontent.com/shivshaktisahoo/test/main/garments_worker_productivity.csv')
garments_dataset

## preprocessing /////////////////
garments_dataset['wip'] = garments_dataset['wip'].fillna(0)
garments_dataset['quarter'] = garments_dataset['quarter'].map({'Quarter1':1, 'Quarter2':2, 'Quarter3':3, 'Quarter4':4, 'Quarter5':5})
for i in range(len(garments_dataset)):
  garments_dataset['department'][i] = garments_dataset['department'][i].strip()
garments_dataset['department'] = garments_dataset['department'].map({'sweing':1, 'finishing':0})
garments_dataset['day'] = garments_dataset['day'].map({'Saturday':1, 'Sunday':2, 'Monday':3, 'Tuesday':4, 'Wednesday':5,'Thursday':6})
var1 = garments_dataset['actual_productivity']
for i in enumerate(var1):
  if var1[i[0]]>0 and var1[i[0]]<=0.4:
    var1[i[0]] = 1
  elif var1[i[0]]>0.4 and var1[i[0]]<=0.75:
    var1[i[0]] = 2
  else:
    var1[i[0]] = 3
garments_dataset

g1 = garments_dataset.drop(['date'], axis=1)
garments_corr = g1.corr()
garments_corr

# relationship between attribute and targeted column with minimum threshold
MIN_THRES = 0.05
garments_final_attr = []
for i in garments_corr.drop(['actual_productivity'],axis=1):
  if abs(garments_corr[i]['actual_productivity']) >= MIN_THRES:
      garments_final_attr.append(i)
garments_final_attr

# relationship between attribute and attribute with maximum threshold
MAX_THRES = 0.85
list1 = list(garments_corr.columns)
max_final_attr = []
for i in enumerate(list1):
  for j in list1[i[0]+1:]:
    if abs(garments_corr[i[1]][j]) >= MAX_THRES:
      print(abs(garments_corr[i[1]][j]),i[1],j)
      max_final_attr.append(i[1])
max_final_attr

X = garments_dataset.drop(['date','actual_productivity','department', 'smv'], axis=1)
Y = garments_dataset['actual_productivity']
neigh = KNeighborsClassifier(n_neighbors=20)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,Y,test_size=0.1)

X

neigh.fit(x_train,y_train)
neigh.score(x_test,y_test)

print(neigh.predict_proba(np.array([1,	6,	8,	0.80,	1108.0,	7080,	98,	0.0,	0,	0,	59.0]).reshape(1,-1)))
print(neigh.predict_proba(np.array([2,	5, 10,	0.75,	0.0,	960,	0,	0.0,	0,	0,	8.0]).reshape(1,-1)))

prediction = neigh.predict_proba(np.array([2,	5, 10,	0.75,	0.0,	960,	0,	0.0,	0,	0,	8.0]).reshape(1,-1))[0]
predicted_target = np.argmax(prediction)
predicted_percentage = round(np.max(prediction)*100,2)
print(f"Their is {predicted_percentage}% chance of target {predicted_target}")

"""- Naive bayes Algorithm"""

from sklearn.naive_bayes import GaussianNB
neigh = GaussianNB()
neigh.fit(x_train, y_train)

prediction = neigh.predict_proba(np.array([2,	5, 10,	0.75,	0.0,	960,	0,	0.0,	0,	0,	8.0]).reshape(1,-1))[0]
predicted_target = np.argmax(prediction)
predicted_percentage = round(np.max(prediction)*100,2)
print(f"Their is {predicted_percentage}% chance of target {predicted_target}")