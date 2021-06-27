import sklearn
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import pickle
import random

garments_dataset = pd.read_csv('garments_worker_productivity.csv')

'''
## preprocessing /////////////////
garments_dataset['wip'] = garments_dataset['wip'].fillna(0)

# garments_dataset.quarter.unique()
# garments_dataset.quarter.describe()
garments_dataset['quarter'] = garments_dataset['quarter'].map({'Quarter1':1, 'Quarter2':2, 'Quarter3':3, 'Quarter4':4, 'Quarter5':5})

# garments_dataset.department.unique()
# garments_dataset.department.describe()
for i in range(len(garments_dataset)):
  garments_dataset['department'][i] = garments_dataset['department'][i].strip()
# garments_dataset.department.unique()
garments_dataset['department'] = garments_dataset['department'].map({'sweing':1, 'finishing':0})

# garments_dataset.day.unique()
garments_dataset['day'] = garments_dataset['day'].map({'Saturday':1, 'Sunday':2, 'Monday':3, 'Tuesday':4, 'Wednesday':5,'Thursday':6})


var1 = garments_dataset['actual_productivity']
for i in enumerate(var1):
  if var1[i[0]]>0 and var1[i[0]]<=0.4:
    var1[i[0]] = 1
  elif var1[i[0]]>0.4 and var1[i[0]]<=0.75:
    var1[i[0]] = 2
  else:
    var1[i[0]] = 3


reg1 = LogisticRegression(random_state=0)
X = garments_dataset.drop(['date','actual_productivity'], axis=1)
for i in range(len(garments_dataset)):
    garments_dataset['actual_productivity'][i] = round(garments_dataset['actual_productivity'][i])
Y = garments_dataset['actual_productivity']



best = 0

for i in range(30):  
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
    reg1.fit(x_train, y_train)  
    acc = reg1.score(x_test, y_test)
    print(f"Accuracy is {round(acc,2)*100}%")
    if acc > best:
        with open('garments_model.pickle', 'wb') as file1:
            pickle.dump(reg1, file1)
        best = acc

print(f"Saved the model with accuracy {round(best,2)*100}%")
'''

pickle_in = open('garments_model.pickle', 'rb')
reg = pickle.load(pickle_in)



# manually taking some inputs

ran_data1 = np.array([[1,0,4,8,0.00,26.16,1108.0,7080,98,0.0,0,0,59.0]])
ran_data2 = np.array([[4,1,2,8,0.10,3.90,0,960,0,0.0,0,0,8.0]])


result1 = reg.predict(ran_data1)[0]
result2 = reg.predict(ran_data2)[0]

# classify the dataset
def result(r):
  if r==1.0:
    return 'Low productivity'
  elif r==2.0:
    return 'Medium productivity'
  else:
    return 'High productivity'

print(f"\nResult1 : {result(result1)}\nResult2 : {result(result2)}")
