'''
Demonstrate cross validation
----------------------------
Although there is library function for cross validation in python, the code 
shows the details so that the students could understand the details
'''

import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split


nFolds = 3 #we are doing 3-fold cross validation

#read the data file
df = pd.read_csv("districtLevelData_discretized.csv")
clf = tree.DecisionTreeClassifier(max_depth=50,criterion='entropy',random_state=0)

#extract the class labels and attributes from the data set
y = df.iloc[:,0]
X = df.iloc[:,1:7] #caution: Python excludes right limit
                   #i.e. we are selecting columns 1 to 6 only 

#calculate the fold length
foldLength = int(df.shape[0]/nFolds)

start_index = 0
for fold in range(0,3):
    end_index = start_index + foldLength
    y_test = df.iloc[start_index:end_index,0]
    X_test = df.iloc[start_index:end_index,0]
    print("%d, %d"%(start_index,end_index))
    start_index = end_index+1