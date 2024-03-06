from pydantic.dataclasses import dataclass
from enum import Enum
from typing import List
import argparse
import logging
from logging.handlers import RotatingFileHandler
# from cosmic.observations.slackbot import SlackBot
import os
from textwrap import dedent
import json
import numpy as np

################################################################################################################
# CONSTANTS - change these to suit your needs:
basebands_8bit = ["AC","BD"]
basebands_3bit = []
SERVICE_NAME = os.path.splitext(os.path.basename(__file__))[0]
CONFIG_HASH = "CAL_configuration"
LOG_HASH = "CAL_log"
GPU_PHASES_REDIS_HASH = "GPU_calibrationPhases"
GPU_GAINS_REDIS_HASH = "GPU_calibrationGains"
GPU_GAINS_REDIS_CHANNEL = "gpu_calibrationgains"
SCAN_END_CHANNEL = "scan_dataset_finish"
OBSERVATIONS_CHANNEL = "observations"
CHANNEL_PRIORITY_ORDER=[OBSERVATIONS_CHANNEL, SCAN_END_CHANNEL, GPU_GAINS_REDIS_CHANNEL]
################################################################################################################

################################################################################################################
# TYPES
class Severity(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    DEBUG = "DEBUG"

@dataclass
class ObservationMetadata:
    # The name of the calibration observation
    obs_id: str = None
    # The frequency centres for the tunings (MHz)
    fcentmhz: List[float]
    # The sideband for the tunings
    sideband:  List[int]
    # The bin time interval for observation
    tbin: float
    # The observation start time as a unix timestamp
    starttime: float
    # The number of frequency channels
    nof_channels: int
    # The number of polarisations
    nof_polarisations: int
    # The number of tunings
    nof_tunings: int
    # The number of streams
    nof_streams: int
    # The project id of the observation
    project_id: str
    # The dataset
    dataset: str
    # bit mode
    bit_mode: int

@dataclass
class CalibrationCollatorConfig:
    # Product directory
    product_dir: str
    # The SNR threshold
    snr_threshold: float
    # Gain receiver timeout (s)
    gain_receiver_timeout: float
    # Collator re-arm time (s)
    re_arm_time: float
    # Fitting method
    fitting_method: str
    # Dry run
    dry_run: bool
    # The runtime mode
    online: bool
    # Should it post to slack
    post_to_slack: bool

@dataclass
class SlackBot:
    name : str

class CalibrationSlackObject:
    def __init__(self, slackbot: SlackBot, message_thread: str):
        self.slackbot = slackbot
        self.message_thread = message_thread

class CalibrationLoggerObject:
        def __init__(self, logger: logging.Logger, severity: Severity):
            self.logger = logger
            self.severity = severity
            self.level = getattr(logging, self.severity.value)

@dataclass
class AntennaGain:
    gain_pol0_real: np.ndarray[float]
    gain_pol0_imag: np.ndarray[float]
    gain_pol1_real: np.ndarray[float]
    gain_pol1_imag: np.ndarray[float]

@dataclass
class Antenna:
    name: str
    flagged_hz : List[float]

@dataclass
class Tuning:
    index: int
    freqs_hz: List[float]
    antenna: List[Antenna]

@dataclass
class CalibrationGainProduct:
    obs_id : str = None
    reference_antenna: str = None
    tunings: List[Tuning] = []
################################################################################################################
    
################################################################################################################
# FUNCTIONS
def get_tuningindex_and_start_frequency(hash_index_key: str) -> (int, int):
    key_split = hash_index_key.split(',')
    tuning = key_split[-1]
    if tuning in basebands_8bit:
        tuning_index = basebands_8bit.index(tuning)
    elif tuning in basebands_3bit:
        tuning_index = basebands_3bit.index(tuning)
    else:
        tuning_index = -1
    return tuning_index

def log_and_post_slackmessage(logger: CalibrationLoggerObject, slackbot: CalibrationSlackObject, message:str,
                               severity:str = "INFO", is_reply:bool = False, update_message:bool = False):
        try:
            level = getattr(logging, severity)
            logger.log(level, message)
            if level == 10: 
                return
        except: 
            logger.error(f"Invalid severity specified: {severity}.")

        if slackbot is not None:   
            if is_reply:
                slackbot.post_message(dedent(f"\
                    {severity}:\n" + message), thread_ts=slack_message_ts)
            elif update_message:
                slackbot.update_message(dedent(f"\
                    {severity}:\n" + message),
                    ts=slack_message_ts,
                )
            else:
                slackbot.post_message(dedent(f"\
                        {severity}:\n" + message))
                slack_message_ts = slackbot.last_message_ts

def parse_gains_contents(input_gains_contents : dict) -> CalibrationGainProduct:
    """
    Called with a set of gains for a particular observation. This function will collate each payload and
    extract required detail.
    """
    #Which antenna have been provided - assume unchanging given timeout expired
    antenna = list(input_gains_contents[list(input_gains_contents.keys())[0]]['gains'].keys())

    calibration_gain_product = CalibrationGainProduct()
    for start_freq_tune, payload in input_gains_contents.items():
        tune_idx = get_tuningindex_and_start_frequency(start_freq_tune)

        #add tuning instance to observation if observation.tunings has tuning index already:
        if any(calibration_gain_product.tunings.index == tuning.index for tuning in calibration_gain_product.tunings):
            calibration_gain_product.tunings.append(Tuning(index = tune_idx, freqs_hz = payload["freqs_hz"], antenna=antenna))
        else:
            calibration_gain_product.tunings = [Tuning(index = tune_idx, freqs_hz = payload["freqs_hz"], antenna=antenna)]
        
        #descend into the payload:



        
    

################################################################################################################

#Offline collation class:
class OfflineCalibrationGainCollator():
    def __init__(self, config: CalibrationCollatorConfig, observation_detail: ObservationDetail):
        self.config = config
        self.observation_detail = observation_detail
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description=("""Listen for updates to GPU hashes containing calibration phases
    and generate residual delays and load calibration phases to the F-Engines.""")
    )
    parser.add_argument('paths', nargs='*')
    args = parser.parse_args()

    input_json_dict = {}
    if len(args.paths) != 0:
        for path in args.paths:
            if os.path.isfile(path):
                with open(path, 'r') as f:
                    input_json_dict.update(json.load(f))
            elif os.path.isdir(path):
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if file.endswith('.json'):
                            file_path = os.path.join(root, file)
                            with open(file_path, 'r') as f:
                                input_json_dict.update(json.load(f))