from cosmic.fengines import ant_remotefeng_map
from cosmic.redis_actions import redis_obj, redis_hget_keyvalues, redis_publish_dict_to_hash, redis_publish_service_pulse
import time
import logging
import math
from logging.handlers import RotatingFileHandler
import redis
import os
import argparse

LOGFILENAME = "/home/cosmic/logs/Delays.log"

SERVICE_NAME = os.path.splitext(os.path.basename(__file__))[0]

logger = logging.getLogger('delay_logger')
logger.setLevel(logging.INFO)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
fh = RotatingFileHandler(LOGFILENAME, mode = 'a', maxBytes = 512, backupCount = 0, encoding = None, delay = False)
fh.setLevel(logging.INFO)

# create formatter
formatter = logging.Formatter("[%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s] %(message)s")

# add formatter to ch
ch.setFormatter(formatter)
fh.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)
logger.addHandler(fh)

class DelayLogger:
    """
    Every period, listen for broadcast delays. Calculate expected loaded delays/phases,
    fetch delays/phases from F-Engine and compare (in conjunction with phase and slope values).

    There are carefully inserted wait statements here to account for delay loading times.
    """
    def __init__(self, redis_obj, polling_rate):
        self.redis_obj = redis_obj
        self.polling_rate = polling_rate
        self.ant_feng_map = ant_remotefeng_map.get_antennaFengineDict(
            self.redis_obj
        )
        
        logger.info("Starting Delay logger...\n")

    def _fetch_model_values(self):
        """
        Fetch the redis delays from META_modelDelays. We don't care about calibration
        delays since we are purely comparing the slope of the delays which 
        are rate dependant
        TODO: predict and verify loaded phase
        
        :return: a mapping of antenna to delay rate values
        :rtype: Dict
        """

        modelDelays = redis_hget_keyvalues(self.redis_obj, "META_modelDelays")
        onesec_future_integer = int(time.time() + 1) #1 second into the future
        ant2delayrate = {}
        for ant, value in modelDelays.items():
            if isinstance(value, dict):
                loadtime_diff_modeltime = (onesec_future_integer - value.get("time_value", 0.0)) #calculate 1s into the future
                delay_raterate = value.get("delay_raterate_nsps2", 0.0)
                delay_rate = value.get("delay_rate_nsps", 0.0)

                # dT/dt = 2ax + b
                model_delay_rate = delay_raterate * 2 * loadtime_diff_modeltime + delay_rate
                logger.info(f"Model delays received for antenna {ant}: {modelDelays[ant]}")
                logger.info(f"Calculated expected delay rate of {model_delay_rate} for antenna {ant}.")
                ant2delayrate[ant] = model_delay_rate
        sleep_time = onesec_future_integer - time.time()
        sleep_time = -1 * sleep_time if sleep_time < 0 else sleep_time
        time.sleep(sleep_time)
        return ant2delayrate
    
    def _fetch_feng_state(self, antname):
        """
        Get the delay state for the F-Engine instance attached to antname.
        
        :param antname: Name of the antenna from which to collect delay state.
        :type antname: str

        :return: dictionary indicating tracking on/off and state of delay tracking
        :rtype: {"on" : bool, "tracking" : bool}
        """
        feng_obj = self.ant_feng_map[antname]
        delay_state = {
            "on" : feng_obj.delay_switch.is_set(),
            "tracking" : feng_obj.delay_track.is_set()
        }
        logger.info(f"Antenna {antname} has delay state {delay_state}.")
        return delay_state

    def _fetch_feng_values(self, antname):
        """
        Given an antenna name, fetch the firmware reported phase
        and slope and check they match with the values received 
        by redis channel.

        :param antname: Name of the antenna from which to collect slope and phase.
        :type antname: str

        :return: firmware slopes and firmware phases of the antenna
        :rtype: tuple - slope,phase
        """
        feng_obj = self.ant_feng_map[antname]
        slope, s_scale = feng_obj.phaserotate.get_firmware_slope(0)
        phase, p_scale = feng_obj.phaserotate.get_firmware_phase(0)
        return slope/s_scale, phase/p_scale    

    def run(self):
        """
        Every polling rate, for every antenna, listen for the 
        delays broadcast, and when received, compare against the loaded values."""
        self.ant2delay_log = {}

        while True:
            redis_publish_service_pulse(self.redis_obj, SERVICE_NAME)
            ant2delayrate = self._fetch_model_values()
            
            for ant, delay_rate in ant2delayrate.items():
                logger.info(f"Checking delay status of antenna {ant}")
                feng_state = self._fetch_feng_state(ant)
                slope, phase = self._fetch_feng_values(ant)
                logger.info(f"Received slope = {slope} and phase = {phase} for antenna {ant}.")

                if feng_state["on"] and feng_state["tracking"]:
                    #thread is live and tracking, check against model
                    logger.info("Delay thread is live and delays are tracking.")
                    is_correct = math.isclose(slope, delay_rate, rel_tol=2**-18)
                    feng_state["correct"] = is_correct
                    if not is_correct:
                        logging.error(f"Firmware reports slope = {slope} which is not close to model value = {delay_rate}.")
                else:
                    logger.info("Delay thread is dead")
                    #calibration mode or delay thread is dead, check against 0
                    is_correct = math.isclose(slope, 0.0, rel_tol=2**-18)
                    feng_state["correct"] = is_correct
                    if not is_correct:
                        logging.error(f"Firmware reports slope = {slope} which is not close to model value = {delay_rate}.")

                self.ant2delay_log[ant] = feng_state
            redis_publish_dict_to_hash(self.redis_obj, "FENG_delayStatus", self.ant2delay_log)

            time.sleep(self.polling_rate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description=("Set up the Delay logger.")
    )
    parser.add_argument(
    "-p","--polling_rate", type=int, help="Rate at which delays are checked.", default=30
    )
    parser.add_argument(
    "-c", "--clean", action="store_true",help="Delete the existing log file and start afresh.",
    )
    args = parser.parse_args()
    if os.path.exists(LOGFILENAME) and args.clean:
        print("Removing previous log file...")
        os.remove(LOGFILENAME)
    else:
        print("Nothing to clean, continuing...")

    feng_logger = DelayLogger(redis_obj, polling_rate = args.polling_rate)
    feng_logger.run()
