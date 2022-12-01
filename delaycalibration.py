import pandas as pd
import numpy as np
import argparse
import astropy.constants as const
import os
from cosmic.redis_actions import redis_obj, redis_publish_dict_to_hash

#CONSTANTS
ADVANCE_TIME = (8e3/(0.66*const.c.value)) #largest baseline 8km / (2/3rds c) ~ largest calibration delay in s

class DelayCalibration():
    def __init__(self, redis_obj, calib_csv):
        """
        A delay calibration calculation object.
        Makes use of observation recordings
        to calculate the calibration delays for each stream
        of each antenna and sends them 
        to the redis hash META_calibrationDelays.
        """
        self.redis_obj = redis_obj

        #initialise calibration delay data dictionary
        self.ant2calibmap_init = pd.read_csv(os.path.abspath(calib_csv), names = ["IF0","IF1","IF2","IF3"], header=None, skiprows=1).to_dict('index')
        self.ant2calibmap = {}

    def run(self):
        """
        This function will take the calibration delay values in conjunction with ADVANCE_TIME
        and publish them to redis hash META_calibrationDelays.
        """
        for ant, calib_value in self.ant2calibmap_init.items():
            values = np.fromiter(calib_value.values(),dtype=float)
            values = values + (ADVANCE_TIME* 1e9)
            tmp_calib_values = {}
            for i, key in enumerate(calib_value):
                tmp_calib_values[key] = values[i]
            self.ant2calibmap[ant] = tmp_calib_values
        self.publish_calibration_delays()

    def publish_calibration_delays(self):
        """
        Push out calibration delay values on individual channels for each antenna.
        Then push out the full dictionary to a hash for display purposes.
        """
        redis_publish_dict_to_hash(self.redis_obj, "META_calibrationDelays", self.ant2calibmap)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description=("Set up the Delay logger.")
    )
    parser.add_argument("calibration_csv", type=str, help="calibration.csv filepath.")
    args = parser.parse_args()
    
    delayCalibration = DelayCalibration(redis_obj, args.calibration_csv)
    delayCalibration.run()