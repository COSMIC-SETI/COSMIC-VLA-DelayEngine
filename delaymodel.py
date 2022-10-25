import numpy as np
import pandas as pd
import os
import logging
import time
import threading
import evla_mcast
from delay_engine.phasing import compute_uvw
import astropy.constants as const
from astropy.coordinates import SkyCoord
from astropy.time import Time
from cosmic.redis_actions import redis_obj, redis_hget_keyvalues, redis_publish_dict_to_hash

#LOGGING
logging.basicConfig(
    filename="/home/cosmic/logs/DelayModel.log",
    format="[%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s] %(message)s",
    level=logging.INFO,
)

logging.getLogger("delaymodel").setLevel(logging.INFO)

#CONSTANTS
WD = os.path.realpath(os.path.dirname(__file__))
DEFAULT_ANT_ITRF = os.path.join(WD, "vla_antenna_itrf.csv")
CLOCK_FREQ = 2.048e9 #Gsps

ADVANCE_TIME = (8e3/const.c.value) #largest baseline 8km / c ~ largest calibration delay in s
NOADVANCE = False
TIME_INTERPOLATION_LENGTH=1 

ITRF_CENTER = [-1601185.4, -5041977.5, 3554875.9] # x,y,z

class DelayModel(evla_mcast.Controller):
    def __init__(self, redis_obj):
        """
        A delay calculation object.
        Makes use of observation metadata and the vla antenna positions
        to calculate the geometric delays for each antenna and publish them
        to redis.
        """
        #Initialise the mcast controller
        evla_mcast.Controller.__init__(self, delays=False, obs_group='239.192.3.12')
        self.scans_require = ["obs", "stop"]  # 'vci', 'ant'

        self.redis_obj = redis_obj
        self.itrf = pd.read_csv(DEFAULT_ANT_ITRF, names=['X', 'Y', 'Z'], header=None, skiprows=1)
        self.antname_inorder = list(self.itrf.index.values)
        self.num_ants = len(self.antname_inorder)
        self.data = {
                    "delay_ns":[0.0]*self.num_ants,
                    "delay_rate_nsps":[0.0]*self.num_ants,
                    "delay_raterate_nsps2":[0.0]*self.num_ants,
                    "time_value":0.0
                    }
        self.init_source_from_redis()
        self.publish_delays()
        self.calculate_delay_thread = threading.Thread(
            target=self.calculate_delay, args=(), daemon=False
        )
        logging.info("Starting delay calculation thread...")
        self.calculate_delay_thread.start()

    def init_source_from_redis(self):
        logging.info(f"Initialize source coordinates...")
        obs_meta = redis_hget_keyvalues(
            self.redis_obj, "META"
        )
        self.ra = obs_meta.get('ra_deg', None)
        self.dec = obs_meta.get('dec_deg', None)
        self.source = SkyCoord(self.ra, self.dec, unit='deg')

    def handle_config(self, scan):
        self.scan = scan
         # This function get the scans with a complete metadata from evla_mcast
        logging.info("Handling updated ra and dec values from observation metadata")

        # Parse phase center coordinates
        self.ra = scan.ra_deg
        self.dec = scan.dec_deg
        self.source = SkyCoord(self.ra, self.dec, unit='deg')
        return True
    
    def calculate_delay(self):
        while True:
            t = np.floor(time.time())
            tts = [3, (TIME_INTERPOLATION_LENGTH/2) + 3, TIME_INTERPOLATION_LENGTH + 3] # Interpolate time samples with 3s advance
            tts = np.array(tts) + t
            dt = TIME_INTERPOLATION_LENGTH/2
            ts = Time(tts, format='unix')

            # perform coordinate transformation to uvw
            uvw1 = compute_uvw(ts[0], self.source, self.itrf[['X','Y','Z']], ITRF_CENTER)
            uvw2 = compute_uvw(ts[1], self.source, self.itrf[['X','Y','Z']], ITRF_CENTER)
            uvw3 = compute_uvw(ts[2], self.source, self.itrf[['X','Y','Z']], ITRF_CENTER)

            # "w" coordinate represents the goemetric delay in light-meters
            w1 = uvw1[...,2]
            w2 = uvw2[...,2]
            w3 = uvw3[...,2]

            # Calibration delays are added in the controller
            delay1 = (w1/const.c.value)
            delay2 = (w2/const.c.value)
            delay3 = (w3/const.c.value)

            delay1 = -delay1
            delay2 = -delay2
            delay3 = -delay3

            # advance all the delays forward in time
            if not NOADVANCE:
                delay1 += ADVANCE_TIME
                delay2 += ADVANCE_TIME
                delay3 += ADVANCE_TIME

            # Compute the delay rate in s/s
            rate1 = (delay2 - delay1) / (dt)
            rate2 = (delay3 - delay2) / (dt)

            # Compute the delay rate rate in s/s^2
            raterate = (rate2 - rate1) / (dt)

            self.data["delay_ns"] = (delay1*1e9).tolist()
            self.data["delay_rate_nsps"] = (rate1*1e9).tolist()
            self.data["delay_raterate_nsps2"] = (raterate*1e9).tolist()
            self.data["time_value"] = t
            logging.debug(f"Calculated the following delay data dictionary: {self.data}")
            self.publish_delays()
            time.sleep(1)
    
    def publish_delays(self):
        df = pd.DataFrame(self.data, index=self.antname_inorder)
        delay_dict = df.to_dict('index')
        logging.info(f"Publishing delays dictionary")
        redis_publish_dict_to_hash(self.redis_obj, "META_modelDelays", delay_dict)

if __name__ == "__main__":
    delayModel = DelayModel(redis_obj)
    delayModel.run()