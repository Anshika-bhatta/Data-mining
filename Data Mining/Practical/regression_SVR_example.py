#This isn't a complete code
#You need to add much more.......!!!

from sklearn.svm import SVR
import numpy as np
import pandas as pd

reg = SVR(C=2,epsilon=0.1)


df = pd.read_csv('Concrete_data.csv')

X = df.iloc[:,0:7].to_numpy()
y = df.iloc[:,8].to_numpy()

reg.fit(X,y)

print(reg.predict(X))

#do yourselves
#permutation test
X_col_permuted = np.random.shuffle(X[:,0])
