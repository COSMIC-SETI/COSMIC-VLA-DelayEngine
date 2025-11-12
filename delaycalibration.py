import pandas as pd
import numpy as np
import argparse
import astropy.constants as const
import time
import os
from cosmic.redis_actions import redis_obj, redis_publish_dict_to_hash, redis_publish_dict_to_channel

#CONSTANTS
ADVANCE_TIME = (8e3/(0.66*const.c.value)) #largest baseline 8km / (2/3rds c) ~ largest calibration delay in s
CALIBRATION_CACHE_HASH = "CAL_fixedValuePaths"

def load_delay_calibrations(calib_csv, fallback_csv=None):
    #read in calibration delay data dictionary
    ant2calibmap_init = pd.read_csv(os.path.abspath(calib_csv), names = ["IF0","IF1","IF2","IF3"], header=None, skiprows=1).to_dict('index')
    ant2calibmap = {}
    for ant, calib_value in ant2calibmap_init.items():
        values = np.fromiter(calib_value.values(),dtype=float)
        values = values + (ADVANCE_TIME* 1e9)
        tmp_calib_values = {}
        for i, key in enumerate(calib_value):
            tmp_calib_values[key] = values[i]
        ant2calibmap[ant] = tmp_calib_values
    
    redis_publish_dict_to_hash(redis_obj, "META_calibrationDelays", ant2calibmap)
    redis_publish_dict_to_channel(redis_obj, "update_calibration_delays", True)
    redis_publish_dict_to_hash(redis_obj, CALIBRATION_CACHE_HASH, {"fixed_delay":calib_csv})

    if fallback_csv is not None:
        using_default = calib_csv == fallback_csv
        redis_publish_dict_to_hash(redis_obj, CALIBRATION_CACHE_HASH, {"using_default_delays": using_default, "time_unix": time.time()})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description=('Collect fixed delays from csv file and publish them to "META_calibrationDelays".')
    )
    parser.add_argument("-f","--fixed-csv", type=str, help="path to the latest fixed-delays csv.")
    args = parser.parse_args()
    
    load_delay_calibrations(os.path.abspath(args.fixed_csv))