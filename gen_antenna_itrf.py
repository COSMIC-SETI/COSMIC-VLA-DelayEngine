import pandas as pd
import numpy as np
from cosmic.redis_actions import redis_obj, redis_hget_keyvalues

ant2propmap = redis_hget_keyvalues(redis_obj, "META_antennaProperties")

listlen = len(ant2propmap)
X = [None]*listlen
Y = [None]*listlen
Z = [None]*listlen
ANTNAMES = [None]*listlen
index = 0
for antname, prop in ant2propmap.items():
    #Select only antenna with positions
    if ('X' in prop) and ('Y' in prop) and ('Z' in prop):
        ANTNAMES[index] = antname
        X[index] = prop['X']
        Y[index] = prop['Y']
        Z[index] = prop['Z']

        data = {"X":X, "Y":Y, "Z":Z}
        df = pd.DataFrame(data, index=ANTNAMES)
        index+=1

df = df[df.index.notnull()]
df.to_csv('vla_antenna_itrf.csv', header=False)
print(df)
