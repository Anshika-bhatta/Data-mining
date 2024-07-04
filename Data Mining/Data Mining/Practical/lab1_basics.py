import pandas as pd

df = pd.read_csv("districtLevelData.csv")

table1 = pd.crosstab(index = df['district'],columns=df['Ecological belt'])

md = df['internet'].median()
internet = pd.Series('Low',range(75))
for i in range(0,75):
    if(df['internet'][i]>md):
        internet[i] = 'High'
table2 = pd.crosstab(index = internet,columns=df['Development Region'])

from scipy import stats
c, p, dof, expected = stats.chi2_contingency(table2)

from scipy.stats import pearsonr
import numpy as np
rho = df.corr()
pval = df.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*rho.shape)

'''
p = pval.applymap(lambda x: ''.join(['*' for t in [.05, .01, .001] if x<=t]))
rho.round(2).astype(str) + p
'''

import matplotlib.pyplot as plt