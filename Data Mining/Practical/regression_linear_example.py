#This isn't a complete code
#You need to add much more.......!!!

import numpy as np

from sklearn.linear_model import LinearRegression


#data
year = np.array([2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021])
gdp = np.array([791,794,809,828,882,880,1028,1162,1186,1139,1208])

year = year.reshape(-1,1)#we require predictors in a form of matri
reg = LinearRegression().fit(year, gdp)



print(reg.score(year,gdp))


