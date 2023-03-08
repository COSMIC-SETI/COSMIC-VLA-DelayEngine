from cosmic.redis_actions import redis_obj, redis_publish_service_pulse, redis_hget_keyvalues, redis_publish_dict_to_hash
from cosmic.fengines import ant_remotefeng_map
from cosmic.fengines import delays, configure
import time
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
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

def fetch_delay_status_dict(redis_obj, ant_feng_map):
    """
    Function collects from Redis hashes and over remoteobjects from the F-Engines a 
    status dictionary that it will return.

    Args:
        redis_obj object: an appropriate redis object that contains relevant hashes
        ant_feng_map dict: mapping of antenna names to cosmic_feng objects

    Returns:
        delay_dict dict: {ant:delay_dict} where delay_dict is a dictionary containing many fields detailing
                        the overall state of the F-Engine tracking, fshift and calibration state
        bad_ant_list: a list of F-Engines that are unreachable
    """
    ant_delay_status_dict = {}
    antenna = list(ant_feng_map.keys())
    delay_status_dict = redis_hget_keyvalues(redis_obj, "FENG_delayStatus", keys = antenna)
    ant_residual_phase_dict = redis_hget_keyvalues(redis_obj, "META_calibrationPhases", keys = antenna)
    antname_lo_fshift_dict = delays.get_antToFshiftMap(
            redis_obj,
            ["A", "B", "C", "D"],
            delays.get_sideband(redis_obj),
            antenna,
        )
    bad_ant_list = []
    for ant, feng in ant_feng_map.items():
        try:
            if ant in delay_status_dict:
                delay_dict = delay_status_dict[ant]
                feng_delay_status = feng.get_status_delay_tracking()
                delay_dict["ok"] = str(feng_delay_status["ok"])
                delay_dict["on"] = str(feng_delay_status["is_alive"])
                delay_dict["tracking"] = feng.get_delay_tracking_mode()
                #Check phase_calibration values against META_residualPhases:
                phase_correct = []
                feng_fshifts = np.round(delay_dict["loaded_fshift_hz"]).tolist()
                expected_fshifts = np.round(configure.order_lo_dict_values(antname_lo_fshift_dict[ant])).tolist()
                if ant in ant_residual_phase_dict:
                    expected_residual_phase = (np.array(ant_residual_phase_dict[ant],dtype=float) + np.pi) % (2 * np.pi) - np.pi
                    for stream in range(4):
                        phase_correct += [bool(np.all(np.isclose(expected_residual_phase[stream,:],
                                        np.array(feng.phaserotate.get_phase_cal(stream),dtype=float), atol=1e-1)))] 
                else:
                    phase_correct += [bool(np.all(np.isclose(np.array([0.0]*1024,dtype=float),
                        np.array(feng.phaserotate.get_phase_cal(stream),dtype=float), atol=1e-1)))] 
                delay_dict["expected_fshift_hz"] = expected_fshifts
                delay_dict["fshifts_correct"] = feng_fshifts == expected_fshifts
                delay_dict["phase_cal_correct"] = phase_correct
                ant_delay_status_dict[ant] = delay_dict
            else:
                 ant_delay_status_dict[ant] =  f"No delay status available for {ant}..."
        except:
            ant_delay_status_dict[ant] = f"Unable to reach {ant}. F-Engine may be unreachable."
            bad_ant_list += [ant]
            continue
    return ant_delay_status_dict, bad_ant_list

class DelayLogger:
    """
    Every period, listen for broadcast delays. Calculate expected loaded delays/phases,
    fetch delays/phases from F-Engine and compare (in conjunction with phase and slope values).

    There are carefully inserted wait statements here to account for delay loading times.
    """
    def __init__(self, redis_obj, polling_rate):
        self.redis_obj = redis_obj
        self.polling_rate = polling_rate
        self.ant_feng_map = ant_remotefeng_map.get_antennaFengineDict(redis_obj)

        logger.info("Starting Delay logger...\n")
    
    def run(self):
        """
        Every polling rate, fetch from fetch_delay_status_dict() an antenna:delaydict mapping 
        to give an indication of the correctness of tracking.
        """
        i = 0
        while True:
            if i > 20 and len(bad_ant_list) != 0:
                self.ant_feng_map = ant_remotefeng_map.get_antennaFengineDict(redis_obj)
                i = 0
            else:
                redis_publish_service_pulse(self.redis_obj, SERVICE_NAME)
                t = time.time()
                ant_delay_status_dict, bad_ant_list = fetch_delay_status_dict(self.redis_obj, self.ant_feng_map)
                duration = time.time() - t
                print(f"Duration = {duration}s")
                logger.info(f"Delay state: {ant_delay_status_dict}")
                redis_publish_dict_to_hash(self.redis_obj, "FENG_delayState", ant_delay_status_dict)
                time.sleep(self.polling_rate - duration if duration < self.polling_rate else 0.0)
                i+=1

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
