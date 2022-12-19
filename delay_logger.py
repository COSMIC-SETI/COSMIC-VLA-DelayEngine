from cosmic.redis_actions import redis_obj, redis_publish_service_pulse, redis_hget_keyvalues, redis_publish_dict_to_hash
from cosmic.fengines import ant_remotefeng_map
import time
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
            for ant, feng in self.ant_feng_map.items():
                if ant in delay_status_dict:
                    delay_dict = delay_status_dict[ant]
                    delay_dict["on"] = str(feng.delay_switch.is_set())
                    if (feng.delay_track.is_set() and not feng.delay_halfoff.is_set() and not feng.delay_halfcal.is_set()
                        and not feng.delay_halfphase.is_set()):
                        delay_dict["tracking"] = "True"
                    elif (feng.delay_track.is_set() and feng.delay_halfcal.is_set()):
                        delay_dict["tracking"] = "half-cal"
                    elif (feng.delay_track.is_set() and feng.delay_halfoff.is_set()):
                        delay_dict["tracking"] = "half-off"
                    elif (feng.delay_track.is_set() and feng.delay_halfphase.is_set()):
                        delay_dict["tracking"] = "half-phase"
                    elif (feng.delay_track.is_set() and feng.delay_halfphasecorrection.is_set()):
                        delay_dict["tracking"] = "half-corrected-phase"
                    else:
                        delay_dict["tracking"] = "fixed-only"

                    new_delay_status_dict[ant] = delay_dict
                else:
                    new_delay_status_dict[ant] = f"No delay status available for {ant}..."

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
