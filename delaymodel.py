import numpy as np
import pandas as pd
import os
import logging
from logging.handlers import RotatingFileHandler
import argparse
import time
import threading
import json
import redis
from delay_engine.phasing import compute_uvw
import astropy.constants as const
from astropy.coordinates import SkyCoord
from astropy.time import Time
from cosmic.redis_actions import redis_obj, redis_hget_keyvalues, redis_publish_service_pulse, redis_publish_dict_to_hash, redis_publish_dict_to_channel

LOGFILENAME = "/home/cosmic/logs/DelayModel.log"

SERVICE_NAME = os.path.splitext(os.path.basename(__file__))[0]

#LOGGER:
logger = logging.getLogger('model_delays')
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
fh = RotatingFileHandler(LOGFILENAME, mode = 'a', maxBytes = 512, backupCount = 0, encoding = None, delay = False)
fh.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter("[%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s] %(message)s")

# add formatter to ch
ch.setFormatter(formatter)
fh.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)
logger.addHandler(fh)

#CONSTANTS
TIME_INTERPOLATION_LENGTH=1 
ITRF_X_OFFSET = -1601185.4
ITRF_Y_OFFSET = -5041977.5
ITRF_Z_OFFSET = 3554875.9
ITRF_CENTER = [ITRF_X_OFFSET, ITRF_Y_OFFSET, ITRF_Z_OFFSET]

class DelayModel(threading.Thread):
    def __init__(self, redis_obj, min_loadtime_offset = 3, max_loadtime_offset = 10):
        """
        A delay calculation object.
        Makes use of observation metadata and the vla antenna positions
        to calculate the geometric delays for each antenna and send them 
        to the fengine nodes via redis channels.

        :param redis_obj: The redis object to connect to for access to relavent redis hashes
        :type redis_obj: class
        """
        threading.Thread.__init__(self)

        self.redis_obj = redis_obj
        self.min_loadtime_offset = min_loadtime_offset #s
        self.max_loadtime_offset = max_loadtime_offset #s

        #intialise source pointings dictionary
        self.source_points_lock = threading.Lock()
        self.source_point_update = threading.Event()
        self.source_pointings_dict = {}

        #initialise the antenna positions
        self.itrf_lock = threading.Lock()
        logger.info("Collecting initial antenna positions...")
        self._calc_itrf_from_antprop(redis_hget_keyvalues(
            self.redis_obj, "META_antennaProperties"
        ))

        #initialise the source coordinates
        logger.info("Collecting initial ra and dec positions...")
        self._calc_source_coords_and_lo_eff_from_obs_meta(redis_hget_keyvalues(
            self.redis_obj, "META"
        ))

        #initialise the ant to fshift dictionary
        self.antname_fshift_lock = threading.Lock()

        #set up delay_data dictionary
        self.delay_data = {
            "delay_ns":None,
            "delay_rate_nsps":None,
            "delay_raterate_nsps2":None,
            "effective_lo_0_mhz":None,
            "effective_lo_1_mhz":None,
            "sideband_0":None,
            "sideband_1":None,
            "time_value":None,
            "loadtime_us":0
            }
        self.last_sent_timestamp = 0.0
        
        #thread for calculating delays
        self.calculate_delay_thread = threading.Thread(
            target=self.calculate_delay, args=(), daemon=False
        )

        #thread for listening to ra/dec, antposition and other updates
        self.listen_for_source = threading.Thread(
            target=self.redis_chan_listener, args=(), daemon=False
        )

    def run(self):
        """
        Start the threads that listen for source coordinate updates and
        calculate fresh delays.
        """
        logger.info("Starting source coord listener...")
        self.listen_for_source.start()
        logger.info("Starting delay calculation thread...")
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
        with  self.itrf_lock:
            self.itrf = df[df.index.notnull()]
            self.antnames = df.index[df.index.notnull()]
            logger.info("Collected antenna position values. Publishing now...")
            redis_publish_dict_to_hash(self.redis_obj, "META_antennaITRF",  self.itrf.to_dict('index'))
    
    def _calc_source_coords_and_lo_eff_from_obs_meta(self, obs_meta):
        """
        Accept a meta obs dictionary and calculate source
        coordinates from the ra/dec contained therein. Store in 
        source_point_dict{loadtime: {'ra': xx, 'dec' : xx, 'src' : source, 'sslo' : [], 'sideband' : []}}
        """
        ra = obs_meta.get('ra_deg')
        dec = obs_meta.get('dec_deg')
        sslo = obs_meta.get('sslo')
        sideband = obs_meta.get('sideband')
        try:
            loadtime = obs_meta.get('loadtime')
        except:
            loadtime = None
            logger.info("No loadtime present in the observation metadata provided.")
        logger.info(f"Collected ra {ra}, dec {dec}, sslo {sslo} and sideband {sideband} from obs_meta.")

        #Update source dictionary with newly collected source
        with self.source_points_lock:
            self.source_pointings_dict = {
                'ra'        : ra,
                'dec'       : dec,
                'sslo'      : sslo,
                'sideband'  : sideband,
                'loadtime'  : loadtime
            }

    def redis_chan_listener(self):
        """This function listens for updates on the  "meta_antennaproperties"
        and "obs_phase_center" redis channels.
        Messages are listened for continuously but their contents is checked in
        calculate delay at a polling rate."""
        pubsub = self.redis_obj.pubsub(ignore_subscribe_messages=True)
        for channel in [
            "meta_antennaproperties",
            "obs_phase_center"
        ]:
            try:
                pubsub.subscribe(channel) 
            except redis.RedisError:
                logger.error(f'Subscription to `{channel}` unsuccessful.')
        for message in pubsub.listen():
            if message is not None and isinstance(message, dict):
                if message['channel'] == "meta_antennaproperties":
                    #Then here we want to collect X,Y and Z coordinates
                    ant2prop = json.loads(message.get('data'))
                    self._calc_itrf_from_antprop(ant2prop)
                if message['channel'] == "obs_phase_center":
                    obsmeta = json.loads(message.get('data'))
                    self._calc_source_coords_and_lo_eff_from_obs_meta(obsmeta)
                    #Tell other processes that a new pointing has arrived
                    self.source_point_update.set()
    
    def calculate_delay(self, publish : bool = True):
        """
        Started in a thread, this function takes `new_delay_period` to generate
        a set of delay coordinates for delay tracking on the F-Engines.
        If a new source pointing arrives that has a loadtime sub `threshold`
        delay coefficients for this pointing will be immediately calculated and
        sent to F-Engines.
        
        Delay coefficients are calculated by taking the updated right ascension and
        declination in conjunction with the antenna positions (in itrf measurements)
        to calculate the geometric delays for each antenna.

        If started manually so that the delay dictionary is returned, this function
        makes use of the redis hash antenna coordinates and source coordinates. Therefore
        for custom coordinates, self.itrf or self.source must be overwritten.
        """
        #initialise:
        with self.source_points_lock:
            sideband = self.source_pointings_dict['sideband']
            sslo = self.source_pointings_dict['sslo']
            self.source =  SkyCoord(self.source_pointings_dict['ra'], self.source_pointings_dict['dec'], unit='deg')

        while True:
            redis_publish_service_pulse(self.redis_obj, SERVICE_NAME)

            #Now, we want to check whether the source pointing has updated OR whether it has been over 5 seconds since
            #we last calculated delay coefficients:
            t = time.time()
            t_us = t*1e6
            if ((t - self.last_sent_timestamp) >= 5 or self.source_point_update.is_set()):
                #clear event here as opposed to at the end of the if block
                #to reduce risk that the listening process sets the event as loadtime becomes > 5s and we 
                #only process the new pointing ~5s from now or when the next pointing arrives.
                self.source_point_update.clear()
                t_int = np.round(t_us) #microsecond integer
                with self.source_points_lock:

                    loadtime_from_now = None if self.source_pointings_dict['loadtime'] is None else (self.source_pointings_dict['loadtime'] - t_us)*1e-6 
                    if loadtime_from_now is None:
                        #loadtime is None. Update source pointing and calculate coefficients and set loadtime to 
                        #self.min_loadtime_offset from now.
                        self.source = SkyCoord(self.source_pointings_dict['ra'], self.source_pointings_dict['dec'], unit='deg')
                        time_to_load = t_int + int(self.min_loadtime_offset*1e6)
                        sideband = self.source_pointings_dict['sideband']
                        sslo = self.source_pointings_dict['sslo']
                    
                    elif (loadtime_from_now < self.min_loadtime_offset
                        or 
                        loadtime_from_now > self.max_loadtime_offset):
                        #loadtime for source pointing is too far in the future or is in the past. Use current pointings on hand and set loadtime to 
                        #self.min_loadtime_offset from now.
                        time_to_load = t_int + int(self.min_loadtime_offset*1e6)

                    elif(loadtime_from_now >= self.min_loadtime_offset
                        and
                        loadtime_from_now <= self.max_loadtime_offset):
                        #loadtime is within threshold. Update source pointing, calculate coefficients and set loadtime to 
                        #loadtime.
                        self.source = SkyCoord(self.source_pointings_dict['ra'], self.source_pointings_dict['dec'], unit='deg')
                        time_to_load = self.source_pointings_dict['loadtime']
                        sideband = self.source_pointings_dict['sideband']
                        sslo = self.source_pointings_dict['sslo']

                    else:
                        logger.error("Invalid loadtime provided")

                tts = [3, (TIME_INTERPOLATION_LENGTH/2) + 3, TIME_INTERPOLATION_LENGTH + 3]
                tts = np.array(tts) + (t_int * 1e-6) # Interpolate time samples with 3s advance
                dt = TIME_INTERPOLATION_LENGTH/2
                ts = Time(tts, format='unix')

                # perform coordinate transformation to uvw
                with self.itrf_lock:
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

                # Compute the delay rate in s/s
                rate1 = (delay2 - delay1) / (dt)
                rate2 = (delay3 - delay2) / (dt)
                rate = (delay3 - delay1) / (2*dt)

                # Compute the delay rate rate in s/s^2
                raterate = (rate2 - rate1) / (dt)

                self.delay_data["delay_ns"] = (delay2*1e9).tolist()
                self.delay_data["delay_rate_nsps"] = (rate*1e9).tolist()
                self.delay_data["delay_raterate_nsps2"] = (raterate*1e9).tolist()
                self.delay_data["effective_lo_0_mhz"] = sslo[0]
                self.delay_data["effective_lo_1_mhz"] = sslo[1]
                self.delay_data["sideband_0"] = sideband[0]
                self.delay_data["sideband_1"] = sideband[1]
                self.delay_data["time_value"] = tts[1]
                self.delay_data["loadtime_us"] = time_to_load

                if publish:
                    self.publish_delays()
                else:
                    self.last_sent_timestamp = time.time()
                    return pd.DataFrame(self.delay_data, index=list(self.itrf.index.values)).to_dict('index')
                      
            time.sleep(1e-1)

    def publish_delays(self):
        """
        Push out delay values on individual channels for each antenna. Then
        push out the full dictionary to a hash for display purposes.
        """
        df = pd.DataFrame(self.delay_data, index=list(self.itrf.index.values))
        delay_dict = df.to_dict('index')
        for ant, delays in delay_dict.items():
            #add in the lo values for 2nd compensation phase tracking
            delay_dict[ant] = delays
            redis_publish_dict_to_channel(self.redis_obj, f"{ant}_delays", delays)
        self.last_sent_timestamp = time.time()
        delay_dict['deg_ra'] = self.source.ra.deg
        delay_dict['deg_dec'] = self.source.dec.deg
        redis_publish_dict_to_hash(self.redis_obj, "META_modelDelays", delay_dict)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
    description=("Set up the Model delay calculator.")
    )
    parser.add_argument(
    "-c", "--clean", action="store_true",help="Delete the existing log file and start afresh.",
    )
    parser.add_argument(
    "--min-time-offset", type=float ,help="The lower time threshold (s) for the loadtime values received by the delay model",
    required=False, default = 3
    )
    parser.add_argument(
    "--max-time-offset", type=float ,help="The upper time threshold for the loadtime values received by the delay model",
    required=False, default = 10
    )
    args = parser.parse_args()
    if os.path.exists(LOGFILENAME) and args.clean:
        logger.info("Removing previous log file...")
        os.remove(LOGFILENAME)
        logger.info("Log file removed.")
    else:
        logger.info("Nothing to clean, continuing...")
    delayModel = DelayModel(redis_obj, min_loadtime_offset = args.min_time_offset, max_loadtime_offset = args.max_time_offset)
    delayModel.run()
