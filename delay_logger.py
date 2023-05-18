from cosmic.redis_actions import redis_obj, redis_publish_service_pulse, redis_hget_keyvalues, redis_publish_dict_to_hash
from cosmic.fengines import ant_remotefeng_map
from cosmic.fengines import delays, configure
import time
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
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
    ant_correct_phase_dict = redis_hget_keyvalues(redis_obj, "META_calPhasesCorrectlyLoaded", keys = antenna)
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
                delay_dict["is_alive"] = str(feng_delay_status["is_alive"])
                delay_dict["tracking"] = feng.get_delay_tracking_mode()
                feng_fshifts = np.round(delay_dict["loaded_fshift_hz"]).tolist()
                expected_fshifts = np.round(configure.order_lo_dict_values(antname_lo_fshift_dict[ant])).tolist()
                delay_dict["expected_fshift_hz"] = expected_fshifts
                delay_dict["fshifts_correct"] = feng_fshifts == expected_fshifts
                delay_dict["phase_cal_correct"] = ant_correct_phase_dict[ant]
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
    def __init__(self, redis_obj, polling_rate, influxdb_token):
        self.redis_obj = redis_obj
        self.polling_rate = polling_rate
        self.bucket = "delays"
        token = influxdb_token
        self.client = InfluxDBClient(url='http://localhost:8086', token=token)
        self.org="seti"
        self.ant_feng_map = ant_remotefeng_map.get_antennaFengineDict(redis_obj)
        logger.info("Starting Delay logger...\n")

    def send_delaydata_to_influx_db(self,delay_status_dict):
        delay_model = redis_hget_keyvalues(redis_obj, "META_modelDelays")
        calib_delays = redis_hget_keyvalues(redis_obj, "META_calibrationDelays")
        write_api = self.client.write_api(write_options=SYNCHRONOUS)
        ra_dec_not_loaded = True
        time_now = time.time_ns()
        for ant, state in delay_status_dict.items():
            is_alive = state['is_alive']
            is_ok = state['ok']
            if ant in calib_delays:
                fixed_delays = np.fromiter(calib_delays[ant].values(),dtype=float)
            else:
                fixed_delays = np.zeros(4)
            if is_alive == "None":
                continue
            #Process delay model contents
            if ant in delay_model:
                timestamp = int(delay_model[ant]["time_value"]*1e9)
                delay_coeff = Point("delay_coeff").tag("ant",ant).field("delay_ns",delay_model[ant]["delay_ns"]).time(timestamp)
                write_api.write(self.bucket,self.org, delay_coeff)
                delay_coeff = Point("delay_coeff").tag("ant",ant).field("delay_rate_nsps",delay_model[ant]["delay_rate_nsps"]).time(timestamp)
                write_api.write(self.bucket,self.org, delay_coeff)
                delay_coeff = Point("delay_coeff").tag("ant",ant).field("delay_raterate_nsps2",delay_model[ant]["delay_raterate_nsps2"]).time(timestamp)
                write_api.write(self.bucket,self.org, delay_coeff)
                if ra_dec_not_loaded:
                    timestamp = int(delay_model[ant]["loadtime_us"]*1000)
                    delay_point = Point("delay_pointing").field("deg_ra",delay_model["deg_ra"]).time(timestamp)
                    write_api.write(self.bucket,self.org, delay_point)
                    delay_point = Point("delay_pointing").field("deg_dec",delay_model["deg_dec"]).time(timestamp)
                    write_api.write(self.bucket,self.org, delay_point)
                    ra_dec_not_loaded = False

            timestamp = int(state["delays_loaded_at"]*1e9)
            delay_state = Point("delay_state").tag("ant",ant).field("tracking_mode",state['tracking']).time(timestamp)
            write_api.write(self.bucket,self.org, delay_state)
            delay_state = Point("delay_state").tag("ant",ant).field("tracking_ok",int(bool(is_ok))).time(timestamp)
            write_api.write(self.bucket,self.org, delay_state)
            delay_state = Point("delay_state").tag("ant",ant).field("tracking_alive",int(bool(is_alive))).time(timestamp)
            write_api.write(self.bucket,self.org, delay_state)
            delay_state = Point("delay_state").tag("ant",ant).field("loadtime_accurate",int(state['loadtime_accurate'])).time(timestamp)
            write_api.write(self.bucket,self.org, delay_state)
            delay_state = Point("delay_state").tag("ant",ant).field("fshifts_correct",int(state['fshifts_correct'])).time(timestamp)
            write_api.write(self.bucket,self.org, delay_state)
            delay_state = Point("delay_state").tag("ant",ant).field("delay_correct",int(all(state['delay_correct']))).time(timestamp)
            write_api.write(self.bucket,self.org, delay_state)
            delay_state = Point("delay_state").tag("ant",ant).field("phase_correct",int(all(state['phase_correct']))).time(timestamp)
            write_api.write(self.bucket,self.org, delay_state)
            delay_state = Point("delay_state").tag("ant",ant).field("phase_cal_correct",int(state['phase_cal_correct'])).time(timestamp)
            write_api.write(self.bucket,self.org, delay_state)
            for stream in range(4):
                value = Point("delay_values").tag("ant",ant).tag("stream",stream).field("firmware_delay_ns",state['firmware_delay_ns'][stream]).time(timestamp)
                write_api.write(self.bucket,self.org, value)
                value = Point("delay_values").tag("ant",ant).tag("stream",stream).field("expected_delay_ns",state['expected_delay_ns'][stream]).time(timestamp)
                write_api.write(self.bucket,self.org, value)
                value = Point("delay_values").tag("ant",ant).tag("stream",stream).field("firmware_phase_rad",state['firmware_phase_rad'][stream]).time(timestamp)
                write_api.write(self.bucket,self.org, value)
                value = Point("delay_values").tag("ant",ant).tag("stream",stream).field("expected_phase_rad",state['expected_phase_rad'][stream]).time(timestamp)
                write_api.write(self.bucket,self.org, value)
                value = Point("delay_values").tag("ant",ant).tag("stream",stream).field("loaded_fshift_hz",state['loaded_fshift_hz'][stream]).time(timestamp)
                write_api.write(self.bucket,self.org, value)
                value = Point("delay_values").tag("ant",ant).tag("stream",stream).field("expected_fshift_hz",state['expected_fshift_hz'][stream]).time(timestamp)
                write_api.write(self.bucket,self.org, value)

                #Load fixed delays contents
                value = Point("fix_delays").tag("ant",ant).tag("steam",stream).field("fixed_delay_ns",fixed_delays[stream]).time(time_now)
                write_api.write(self.bucket,self.org, value)
                    
            for tune in range(2):
                value = Point("tune_values").tag("ant",ant).tag("tune",tune).field("current_sslo",state['current_sslo'][tune]).time(timestamp)
                write_api.write(self.bucket,self.org, value)
                value = Point("tune_values").tag("ant",ant).tag("tune",tune).field("current_sideband",state['current_sideband'][tune]).time(timestamp)
                write_api.write(self.bucket,self.org, value)
                
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
                redis_publish_dict_to_hash(self.redis_obj, "FENG_delayState", ant_delay_status_dict)
                self.send_delaydata_to_influx_db(ant_delay_status_dict)
                duration = time.time() - t
                time.sleep(self.polling_rate - duration if duration < self.polling_rate else 0.0)
                i+=1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description=("Set up the Delay logger.")
    )
    parser.add_argument(
    "-p","--polling_rate", type=int, help="Rate at which delays are checked.", default=10
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

    if "INFLUXDB_TOKEN" in os.environ:
        influxdb_token = os.environ["INFLUXDB_TOKEN"]

    feng_logger = DelayLogger(redis_obj, args.polling_rate, influxdb_token)
    feng_logger.run()
