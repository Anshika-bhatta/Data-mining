'''
Classifier for boolean data
Data Description: 0 and 1 indicate low and high values of the variables
**************************
Note: You will need the file `districtLevelData_discretized.csv'
'''

import pandas as pd
from sklearn import tree


#read the data file
df = pd.read_csv("C:/Users/lenovo/Desktop/data mining/Data Mining/Practical/districtLevelData_discretized.csv")

#extract the class labels and attributes from the data set
y = df.iloc[:,0]
X = df.iloc[:,1:7] #caution: Python excludes right limit
                   #i.e. we are selecting columns 1 to 6 only 
X = X.astype('category')

#fit the decision tree
clf = tree.DecisionTreeClassifier(max_depth=2,min_samples_split=10,criterion='entropy',random_state=0)
clf.fit(X,y)

#plot the tree for visualization

tree.plot_tree(clf,fontsize=15,feature_names=X.columns,class_names=['Low','High'])

#Compute and display the confusion matrix
y_predicted = clf.predict(X)
r = pd.crosstab(y,y_predicted)
print('\nThe confusion matrix\n:')
print(r)

#print accuracy
r = r.to_numpy()
acc = r.trace()/r.sum()
print('Accuracy: ',acc)

from sklearn.tree import export_text
text = export_text(clf)
print(text)

