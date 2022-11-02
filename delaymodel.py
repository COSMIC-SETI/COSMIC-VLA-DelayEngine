import numpy as np
import pandas as pd
import os
import logging
import time
import threading
import json
import redis
from delay_engine.phasing import compute_uvw
import astropy.constants as const
from astropy.coordinates import SkyCoord
from astropy.time import Time
from cosmic.redis_actions import redis_obj, redis_hget_keyvalues, redis_publish_service_pulse, redis_publish_dict_to_hash, redis_publish_dict_to_channel

#LOGGING
logging.basicConfig(
    filename="/home/cosmic/logs/DelayModel.log",
    format="[%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s] %(message)s",
    level=logging.INFO,
)

SERVICE_NAME = os.path.splitext(os.path.basename(__file__))[0]

logging.getLogger("delaymodel").setLevel(logging.INFO)

#CONSTANTS
TIME_INTERPOLATION_LENGTH=1 
ITRF_X_OFFSET = -1601185.4
ITRF_Y_OFFSET = -5041977.5
ITRF_Z_OFFSET = 3554875.9
ITRF_CENTER = [ITRF_X_OFFSET, ITRF_Y_OFFSET, ITRF_Z_OFFSET]

class DelayModel(threading.Thread):
    def __init__(self, redis_obj):
        """
        A delay calculation object.
        Makes use of observation metadata and the vla antenna positions
        to calculate the geometric delays for each antenna and send them 
        to the fengine nodes via redis channels.
        """
        threading.Thread.__init__(self)

        self.redis_obj = redis_obj

        #initialise the antenna positions
        self._calc_itrf_from_antprop(redis_hget_keyvalues(
            redis_obj, "META_antennaProperties"
        ))

        #initialise the source coordinates
        self._calc_source_coords_from_radec(redis_hget_keyvalues(
            self.redis_obj, "META"
        ))

        #set up delay_data dictionary
        self.delay_data = {
            "delay_ns":None,
            "delay_rate_nsps":None,
            "delay_raterate_nsps2":None,
            "time_value":None
            }
        
        #thread for calculating delays
        self.calculate_delay_thread = threading.Thread(
            target=self.calculate_delay, args=(), daemon=False
        )

        #thread for listening to ra/dec updates
        self.listen_for_source = threading.Thread(
            target=self.redis_chan_listener, args=(), daemon=False
        )

        logging.info("Starting source coord listener...")
        self.listen_for_source.start()
        logging.info("Starting delay calculation thread...")
        self.calculate_delay_thread.start()

    def _calc_itrf_from_antprop(self,ant2propmap):
        """
        Accept an antennaproperty dictionary and return 
        antenna to itrf mapping dictionary:
        {antname: {'X': 00, 'Y': 00, 'Z': 00}}
        """
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
        self.itrf = df[df.index.notnull()]
        logging.debug(f"Received new antenna position values. Publishing now...")
        redis_publish_dict_to_hash(self.redis_obj, "META_antennaITRF",  self.itrf.to_dict('index'))
    
    def _calc_source_coords_from_radec(self, obs_meta):
        """
        Accept an meta obs dictionary and calculate source
        coordinates from the ra/dec contained therein
        """
        self.ra = obs_meta.get('ra_deg')
        self.dec = obs_meta.get('dec_deg')
        self.source = SkyCoord(self.ra, self.dec, unit='deg')
        logging.debug(f"Received new ra value: {self.ra} and dec value {self.dec}")

    def redis_chan_listener(self):
        """This function listens for updates on the  "meta_antennaproperties"
        and "meta_obs" redis channels"""
        pubsub = self.redis_obj.pubsub(ignore_subscribe_messages=True)
        for channel in [
            "meta_antennaproperties",
            "meta_obs"
        ]:
            try:
                pubsub.subscribe(channel) 
            except redis.RedisError:
                logging.error(f'Subscription to `{channel}` unsuccessful.')
        for message in pubsub.listen():
            if message is not None and isinstance(message, dict):
                if message['channel'] == "meta_antennaproperties":
                    #Then here we want to collect X,Y and Z coordinates
                    ant2prop = json.loads(message.get('data'))
                    self._calc_itrf_from_antprop(ant2prop)
                if message['channel'] == "meta_obs":
                    obsmeta = json.loads(message.get('data'))
                    self._calc_source_coords_from_radec(obsmeta)    
    
    def calculate_delay(self):
        """
        Started in a thread, this function will take the updated right ascension and
        declination in conjunction with the antenna positions (in itrf measurements)
        to calculate the geometric delays for each antenna.
        From these delay values we can produce delay rate and delay rate rate values
        so that the fengines may interpolate delays while waiting for new delay value
        updates.
        """
        while True:
            redis_publish_service_pulse(self.redis_obj, SERVICE_NAME)
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

            # Compute the delay rate in s/s
            rate1 = (delay2 - delay1) / (dt)
            rate2 = (delay3 - delay2) / (dt)

            # Compute the delay rate rate in s/s^2
            raterate = (rate2 - rate1) / (dt)

            self.delay_data["delay_ns"] = (delay1*1e9).tolist()
            self.delay_data["delay_rate_nsps"] = (rate1*1e9).tolist()
            self.delay_data["delay_raterate_nsps2"] = (raterate*1e9).tolist()
            self.delay_data["time_value"] = t
            logging.debug(f"Calculated the following delay data dictionary: {self.delay_data}")
            self.publish_delays()
            time.sleep(5)
    
    def publish_delays(self):
        """
        Push out delay values on individual channels for each antenna. Then
        push out the full dictionary to a hash for display purposes.
        """
        df = pd.DataFrame(self.delay_data, index=list(self.itrf.index.values))
        delay_dict = df.to_dict('index')
        for ant, delays in delay_dict.items():
            redis_publish_dict_to_channel(self.redis_obj, f"{ant}_delays", delays)
        delay_dict['ra'] = self.ra
        delay_dict['dec'] = self.dec
        logging.debug(f"Publishing delays dictionary: {delay_dict}")
        redis_publish_dict_to_hash(self.redis_obj, "META_modelDelays", delay_dict)

if __name__ == "__main__":
    delayModel = DelayModel(redis_obj)
    delayModel.run()