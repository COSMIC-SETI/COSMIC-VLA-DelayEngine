import pandas as pd
import threading
import time
import logging
import os
from cosmic.redis_actions import redis_obj, redis_publish_service_pulse, redis_publish_dict_to_hash

#LOGGING
logging.basicConfig(
    filename="/home/cosmic/logs/DelayModel.log",
    format="[%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s] %(message)s",
    level=logging.INFO,
)

logging.getLogger("delaymodel").setLevel(logging.INFO)

SERVICE_NAME = os.path.splitext(os.path.basename(__file__))[0]

class DelayCalibration(threading.Thread):
    def __init__(self, redis_obj):
        """
        A delay calibration calculation object.
        Makes use of observation recordings
        to calculate the calibration delays for each stream
        of each antenna and sends them 
        to the fengine nodes via redis channels.
        """
        threading.Thread.__init__(self)

        self.redis_obj = redis_obj

        #initialise calibration delay data dictionary
        self.ant2calibmap = pd.read_csv("dave_calibration_delays.csv", names = ["IF0","IF1","IF2","IF3"], header=None, skiprows=1).to_dict('index')

        #thread for calculating delays
        self.calculate_calibration_delay_thread = threading.Thread(
            target=self.calc_calibration_delays, args=(), daemon=False
        )

        logging.info("Starting calibration delay thread...")
        self.calculate_calibration_delay_thread.start()

    def calc_calibration_delays(self):
        """
        Started in a thread, this function will take the calibration delay values.
        These values add/subtract from the delay values on the F-Engine.
        """
        while True:
            redis_publish_service_pulse(self.redis_obj, SERVICE_NAME)
            self.publish_calibration_delays()
            time.sleep(60)

    def publish_calibration_delays(self):
        """
        Push out calibration delay values on individual channels for each antenna.
        Then push out the full dictionary to a hash for display purposes.
        """
        for ant, calib_delay in self.ant2calibmap.items():
            self.redis_obj.publish(f"{ant}_calibration_delays",str(calib_delay))
        redis_publish_dict_to_hash(self.redis_obj, "META_calibrationDelays", self.ant2calibmap)

if __name__ == "__main__":
    delayCalibration = DelayCalibration(redis_obj)
    delayCalibration.run()