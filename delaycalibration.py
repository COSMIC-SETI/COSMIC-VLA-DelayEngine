import pandas as pd
import numpy as np
import threading
import time
import logging
from logging.handlers import RotatingFileHandler
import argparse
import astropy.constants as const
import os
from cosmic.redis_actions import redis_obj, redis_publish_service_pulse, redis_publish_dict_to_hash, redis_publish_dict_to_channel

#LOGGING
LOGFILENAME = "/home/cosmic/logs/DelayCalibrations.log"

SERVICE_NAME = os.path.splitext(os.path.basename(__file__))[0]

#LOGGER:
logger = logging.getLogger('calibration_delays')
logger.setLevel(logging.INFO)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
fh = RotatingFileHandler(LOGFILENAME, mode = 'a', maxBytes = 512, backupCount = 1, encoding = None, delay = False)
fh.setLevel(logging.INFO)

# create formatter
formatter = logging.Formatter("[%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s] %(message)s")

# add formatter to ch
ch.setFormatter(formatter)
fh.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)
logger.addHandler(fh)

#CONSTANTS
ADVANCE_TIME = (8e3/const.c.value) #largest baseline 8km / c ~ largest calibration delay in s

class DelayCalibration(threading.Thread):
    def __init__(self, redis_obj, calib_csv, polling_rate):
        """
        A delay calibration calculation object.
        Makes use of observation recordings
        to calculate the calibration delays for each stream
        of each antenna and sends them 
        to the fengine nodes via redis channels.
        """
        threading.Thread.__init__(self)

        self.redis_obj = redis_obj
        self.polling_rate = polling_rate

        #initialise calibration delay data dictionary
        self.ant2calibmap_init = pd.read_csv(os.path.abspath(calib_csv), names = ["IF0","IF1","IF2","IF3"], header=None, skiprows=1).to_dict('index')
        logger.info(f"Read in the following calibration dict:\n{self.ant2calibmap_init}")
        self.ant2calibmap = {}

    def run(self):
        """
        Started in a thread, this function will take the calibration delay values.
        These values add/subtract from the delay values on the F-Engine.
        """
        while True:
            for ant, calib_value in self.ant2calibmap_init.items():
                values = np.fromiter(calib_value.values(),dtype=float)
                values = values + (ADVANCE_TIME* 1e9)
                tmp_calib_values = {}
                for i, key in enumerate(calib_value):
                    tmp_calib_values[key] = values[i]
                self.ant2calibmap[ant] = tmp_calib_values
            redis_publish_service_pulse(self.redis_obj, SERVICE_NAME)
            self.publish_calibration_delays()
            time.sleep(self.polling_rate)

    def publish_calibration_delays(self):
        """
        Push out calibration delay values on individual channels for each antenna.
        Then push out the full dictionary to a hash for display purposes.
        """
        for ant, calib_delay in self.ant2calibmap.items():
            redis_publish_dict_to_channel(self.redis_obj, f"{ant}_calibration_delays", calib_delay)
        logger.info(f"Published calibration delay dictionary:\n{self.ant2calibmap}")
        redis_publish_dict_to_hash(self.redis_obj, "META_calibrationDelays", self.ant2calibmap)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description=("Set up the Delay logger.")
    )
    parser.add_argument("calibration_csv", type=str, help="calibration.csv filepath.")
    parser.add_argument(
    "-p","--polling_rate", type=float, help="Rate at which delays are publish.", default=5
    )
    parser.add_argument(
    "-c", "--clean", action="store_true",help="Delete the existing log file and start afresh.",
    )
    args = parser.parse_args()
    if os.path.exists(LOGFILENAME) and args.clean:
        logging.info("Removing previous log file...")
        os.remove(LOGFILENAME)
    else:
        logging.info("Nothing to clean, continuing...")
    delayCalibration = DelayCalibration(redis_obj, args.calibration_csv, args.polling_rate)
    logger.info("Calibration delay object instantiated, running now...")
    delayCalibration.run()