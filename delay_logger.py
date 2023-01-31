from cosmic.redis_actions import redis_obj, redis_publish_service_pulse, redis_hget_keyvalues, redis_publish_dict_to_hash
from cosmic.fengines import ant_remotefeng_map
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

class DelayLogger:
    """
    Every period, listen for broadcast delays. Calculate expected loaded delays/phases,
    fetch delays/phases from F-Engine and compare (in conjunction with phase and slope values).

    There are carefully inserted wait statements here to account for delay loading times.
    """
    def __init__(self, redis_obj, polling_rate):
        self.redis_obj = redis_obj
        self.polling_rate = polling_rate
        self.ant_feng_map = ant_remotefeng_map.get_antennaFengineDict(self.redis_obj)
        
        logger.info("Starting Delay logger...\n")

    def run(self):
        """
        Every polling rate, fetch from FENG_delayStatus the entire dict
        and ammend it to account for the state of the delay threads before publishing.
        """
        new_delay_status_dict = {}
        while True:
            redis_publish_service_pulse(self.redis_obj, SERVICE_NAME)
            delay_status_dict = redis_hget_keyvalues(self.redis_obj, "FENG_delayStatus")
            ant_residual_phase_dict = redis_hget_keyvalues(self.redis_obj, "META_calibrationPhases")
            for ant, feng in self.ant_feng_map.items():
                try:
                    if ant in delay_status_dict:
                        delay_dict = delay_status_dict[ant]
                        feng_delay_status = feng.get_status_delay_tracking()
                        delay_dict["ok"] = str(feng_delay_status["ok"])
                        delay_dict["on"] = str(feng_delay_status["is_alive"])
                        if (feng.delay_track.is_set() and not feng.delay_halfoff.is_set() and not feng.delay_halfcal.is_set()
                            and not feng.delay_halfphase.is_set() and not feng.delay_halfphasecorrection.is_set()):
                            delay_dict["tracking"] = "True"
                        elif (feng.delay_track.is_set() and feng.delay_halfcal.is_set()):
                            delay_dict["tracking"] = "half-cal"
                        elif (feng.delay_track.is_set() and feng.delay_halfoff.is_set()):
                            delay_dict["tracking"] = "half-off"
                        elif (feng.delay_track.is_set() and feng.delay_halfphase.is_set()):
                            delay_dict["tracking"] = "half-phase"
                        elif (feng.delay_track.is_set() and feng.delay_halfphasecorrection.is_set()):
                            delay_dict["tracking"] = "half-corrected-phase"
                        elif (feng.delay_track.is_set() and feng.delay_fullphasecorrection.is_set()):
                            delay_dict["tracking"] = "full-corrected-phase"
                        else:
                            delay_dict["tracking"] = "fixed-only"


                        #Check phase_calibration values against META_residualPhases:
                        phase_correct = []
                        for stream in range(4):
                            if ant in ant_residual_phase_dict:
                                expected_residual_phase = (np.array(ant_residual_phase_dict[ant][stream],dtype=float) + np.pi) % (2 * np.pi) - np.pi
                                phase_correct += [bool(np.all(np.isclose(expected_residual_phase,
                                                    np.array(feng.phaserotate.get_phase_cal(stream),dtype=float), atol=1e-1)))] 
                            else:
                                phase_correct += [bool(np.all(np.isclose(np.array([0.0]*1024,dtype=float),
                                                    np.array(feng.phaserotate.get_phase_cal(stream),dtype=float), atol=1e-1)))] 

                        delay_dict["phase_cal_correct"] = phase_correct
                        new_delay_status_dict[ant] = delay_dict
                    else:
                        new_delay_status_dict[ant] = f"No delay status available for {ant}..."
                except:
                    print(f"Could not reach antenna {ant}")
                    new_delay_status_dict[ant] = f"Unable to reach {ant}. F-Engine may be unreachable."
                    self.ant_feng_map = ant_remotefeng_map.get_antennaFengineDict(self.redis_obj)
                    continue

            logger.info(f"Delay state: {new_delay_status_dict}")
            redis_publish_dict_to_hash(self.redis_obj, "FENG_delayState", new_delay_status_dict)

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
