import sklearn
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import pickle
import random

ilp_dataset = pd.read_csv('indian_liver_patient.csv')

"""
ilp_dataset['Albumin_and_Globulin_Ratio'] = ilp_dataset['Albumin_and_Globulin_Ratio'].fillna(round(ilp_dataset.Albumin_and_Globulin_Ratio.describe()['mean'], 2))
ilp_dataset['Gender'] = ilp_dataset['Gender'].map({"Male":1, "Female":0})

reg = LogisticRegression(random_state=0)
X = ilp_dataset.drop(['Dataset'], axis=1)
Y = ilp_dataset['Dataset']
best = 0
for i in range(100): 
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, Y, test_size=0.1)
    reg.fit(x_train, y_train) 
    acc = reg.score(x_test, y_test)
    print(f"Accuracy is {round(acc,2)*100}%")
    if acc > best:
        with open('ilp_model.pickle', 'wb') as file1:
            pickle.dump(reg, file1)
        best = acc

print(f"Saved the model with accuracy {round(best,2)*100}%")
"""
pickle_in = open('ilp_model.pickle', 'rb')
reg = pickle.load(pickle_in)



# manually taking some inputs

ran_data1 = np.array([[20,0,0.7,0.1,187,16,18,6.8,3.3,0.9]])
ran_data2 = np.array([[90,	0, 115.47, 54.6,7560, 6601, 1980, 19.1, 45.9,	10.3]])

result1 = reg.predict(ran_data1)[0]
result2 = reg.predict(ran_data2)[0]

# classify the dataset
def result(r):
  if r==1:
    return 'Liver Dieases'
  else:
    return 'No Liver Dieases'

print(f"Result1 : {result(result1)}\nResult2 : {result(result2)}")

# inputs: [[70,	1, 8.4, 4.5	,356, 56, 76, 7.9, 3.7,	0.8]]
# [[90,	0, 5.4, 5.4	,756, 66, 98, 8.1, 4.7,	0.3]]