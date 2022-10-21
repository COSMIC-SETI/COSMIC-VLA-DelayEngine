"""
Collect local X, Y, and Z coordinates from MCAST and convert to
a csv table of itrf values per antenna.
"""

import pandas as pd
import numpy as np
from cosmic.redis_actions import redis_obj, redis_hget_keyvalues, redis_publish_dict_to_hash

ITRF_X_OFFSET = -1601185.4
ITRF_Y_OFFSET = -5041977.5
ITRF_Z_OFFSET = 3554875.9

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
        X[index] = prop['X'] + ITRF_X_OFFSET
        Y[index] = prop['Y'] + ITRF_Y_OFFSET
        Z[index] = prop['Z'] + ITRF_Z_OFFSET
        index+=1

data = {"X":X, "Y":Y, "Z":Z}
df = pd.DataFrame(data, index=ANTNAMES)
df = df[df.index.notnull()]
redis_publish_dict_to_hash(redis_obj, "META_antennaITRF", df.to_dict('index'))
df.to_csv('vla_antenna_itrf.csv', header=False)
