import numpy as np
import pandas as pd
import os
import logging
import time
import threading
import evla_mcast
from delay_engine.phasing import compute_uvw, compute_antenna_gainphase
import astropy.constants as const
from astropy.coordinates import ITRS, SkyCoord, AltAz, EarthLocation
from astropy.time import Time,TimeDelta
from cosmic.redis_actions import redis_obj, redis_hget_keyvalues, redis_publish_dict_to_hash

#LOGGING
logging.basicConfig(
    filename="/home/cosmic/logs/DelayCalculator.log",
    format="[%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s] %(message)s",
    level=logging.INFO,
)

logging.getLogger("delaycalculator").setLevel(logging.DEBUG)

#CONSTANTS
FIX_DELAY = 0.0
WD = os.path.realpath(os.path.dirname(__file__))
DEFAULT_ANT_ITRF = os.path.join(WD, "vla_antenna_itrf.csv")
MAX_SAMP_DELAY = 16384
CLOCK_FREQ = 2.048e9 #Gsps
ADC_SAMP_TIME = 1/CLOCK_FREQ
MAX_DELAY = MAX_SAMP_DELAY * ADC_SAMP_TIME #seconds
ADVANCE_TIME = MAX_DELAY/2
REFANT = "ea01"
NOADVANCE = True
LO = 1.0

class DelayCalculator(evla_mcast.Controller):
    def __init__(self, redis_obj):
        """
        A delay calculation object.
        Makes use of observation metadata and the vla antenna positions
        to calculate the geometric delays for each antenna and publish them
        to redis.
        """
        threading.Thread.__init__(self)
        self.redis_obj = redis_obj
        self.itrf = pd.read_csv(DEFAULT_ANT_ITRF, names=['X', 'Y', 'Z'], header=None, skiprows=1)
        self.irefant = self.itrf.index.values.tolist().index(REFANT)
        self.antname_inorder = list(self.itrf.index.values)
        self.num_ants = len(self.antname_inorder)
        self.data = {"delay_ns": [0.0]*self.num_ants,
                    "delay_rate_nsps":[0.0]*self.num_ants,
                    "phase_rad":[0.0]*self.num_ants,
                    "phase_rate_radps":[0.0]*self.num_ants}
        self.init_source_from_redis()
        self.publish_delays()
        self.calculate_delay_thread = threading.Thread(
            target=self.calculate_delay, args=(), daemon=False
        )
        print("Starting delay calculation thread...")
        self.calculate_delay_thread.start()

    def init_source_from_redis(self):
        print(f"Initialize source coordinates...")
        obs_meta = redis_hget_keyvalues(
            self.redis_obj, "META"
        )
        self.ra = obs_meta.get('ra_deg', None)
        self.dec = obs_meta.get('dec_deg', None)
        self.source = SkyCoord(self.ra, self.dec, unit='deg')

    def handle_config(self, scan):
        self.scan = scan
         # This function get the scans with a complete metadata from evla_mcast
        print("Handling scans with complete metadata")

        # Parse phase center coordinates
        self.ra = scan.ra_deg
        self.dec = scan.dec_deg
        self.source = SkyCoord(self.ra, self.dec, unit='deg')

        return True
    
    def calculate_delay(self):
        while True:
            t = np.floor(time.time())
            tts = [3, 20+3] # Interpolate between t=3 sec and t=20 sec
            tts = np.array(tts) + t

            ts = Time(tts, format='unix')

            # perform coordinate transformation to uvw
            uvw1 = compute_uvw(ts[0],  self.source, self.itrf[['X','Y','Z']], self.itrf[['X','Y','Z']].values[self.irefant])
            uvw2 = compute_uvw(ts[-1], self.source, self.itrf[['X','Y','Z']], self.itrf[['X','Y','Z']].values[self.irefant])

            # "w" coordinate represents the goemetric delay in light-meters
            w1 = uvw1[...,2]
            w2 = uvw2[...,2]

            # Add fixed delays + convert to seconds
            delay1 = FIX_DELAY + (w1/const.c.value)
            delay2 = FIX_DELAY + (w2/const.c.value)

            delay1 = -delay1
            delay2 = -delay2

            # advance all the B-engines forward in time
            if not NOADVANCE:
                delay1 += ADVANCE_TIME
                delay2 += ADVANCE_TIME

            # Compute the delay rate in s/s
            rate = (delay2 - delay1) / (tts[-1] - tts[0])

            phase      = -2 * np.pi * LO*1e6 * delay1
            phase_rate = -2 * np.pi * LO*1e6 * rate

            self.data["delay_ns"] = (delay1*1e9).tolist()
            self.data["delay_rate_nsps"] = (rate*1e9).tolist()
            self.data["phase_rad"] = phase.tolist()
            self.data["phase_rate_radps"] = phase_rate.tolist()

            print(f"Calculated the following delay data dictionary: {self.data}")
            self.publish_delays()
            time.sleep(10)
    
    def publish_delays(self):
        df = pd.DataFrame(self.data, index=self.antname_inorder)
        delay_dict = df.to_dict('index')
        print(f"Publishing dictionary {delay_dict}...")
        redis_publish_dict_to_hash(self.redis_obj, "META_calcDelays", delay_dict)

    def stop(self):
        if self.calculate_delay_thread.is_alive():
            self.calculate_delay_thread.join()

if __name__ == "__main__":
    delayCalculator = DelayCalculator(redis_obj)
    delayCalculator.run()
    delayCalculator.stop() 