import pandas as pd
import numpy as nm
df = pd.read_csv("D:/Data Mining/Data Mining/Practical/districtLevelData.csv")
df.head()
mn= df['Adult literacy']
# nm.mean(mn)
# nm.median(mn)
nm.mode(mn)