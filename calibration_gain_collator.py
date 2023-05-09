import pandas as pd
import numpy as np
import itertools
import argparse
import os
import sys
import atexit
import traceback
import logging
from logging.handlers import RotatingFileHandler
import redis
import time
import json
from delaycalibration import load_delay_calibrations, CALIBRATION_CACHE_HASH
from phasecalibration import load_phase_calibrations
from textwrap import dedent
import pprint
from calibration_residual_kernals import calc_residuals_from_polyfit, calc_residuals_from_ifft, calc_calibration_ant_grade, calc_calibration_freq_grade, calc_full_grade
from cosmic.observations.slackbot import SlackBot
from cosmic.fengines import ant_remotefeng_map
from cosmic.redis_actions import redis_obj, redis_hget_keyvalues, redis_publish_dict_to_hash, redis_clear_hash_contents, redis_publish_service_pulse, redis_publish_dict_to_channel
from plot_delay_phase import plot_delay_phase, plot_gain_phase, plot_gain_amplitude, plot_snr_and_phase_spread, plot_gain_grade

LOGFILENAME = "/home/cosmic/logs/DelayCalibration.log"
logger = logging.getLogger('calibration_delays')
logger.setLevel(logging.DEBUG)

SERVICE_NAME = os.path.splitext(os.path.basename(__file__))[0]

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

CONFIG_HASH = "CAL_configuration"

GPU_PHASES_REDIS_HASH = "GPU_calibrationPhases"
GPU_GAINS_REDIS_HASH = "GPU_calibrationGains"

GPU_GAINS_REDIS_CHANNEL = "gpu_calibrationgains"
SCAN_END_CHANNEL = "scan_dataset_finish"
OBSERVATIONS_CHANNEL = "observations"

CHANNEL_ORDER=[OBSERVATIONS_CHANNEL, SCAN_END_CHANNEL, GPU_GAINS_REDIS_CHANNEL]

CHANNEL_MESSAGES={OBSERVATIONS_CHANNEL:[],
                  SCAN_END_CHANNEL:[],
                  GPU_GAINS_REDIS_CHANNEL:[]}

class CalibrationGainCollector():
    def __init__(self, redis_obj, fetch_config = False, user_output_dir='.', hash_timeout=20, re_arm_time = 30, fit_method = "fourier", dry_run = False,
    nof_streams = 4, nof_tunings = 2, nof_pols = 2, nof_channels = 1024, slackbot=None, input_fixed_delays = None, input_fixed_phases = None,
    input_json_dict = None, input_fcents = None, input_sideband = None, input_tbin = None, snr_threshold = 4.0):
        self.redis_obj = redis_obj
        self.user_output_dir = user_output_dir
        self.hash_timeout = hash_timeout
        self.re_arm_time = re_arm_time
        self.fit_method = fit_method
        self.dry_run = dry_run
        self.slackbot = slackbot
        self.slack_message_ts = None
        self.input_fixed_delays = input_fixed_delays
        self.input_fixed_phases = input_fixed_phases
        self.input_json_dict = input_json_dict
        self.fcents = input_fcents
        self.input_sideband = input_sideband
        self.tbin = input_tbin
        self.snr_threshold = snr_threshold
        self.nof_streams = nof_streams
        self.nof_channels = nof_channels
        self.nof_tunings = nof_tunings
        self.nof_pols = nof_pols
        self.projid = "None"
        self.dataset = "None"   
        self.scan_is_ending=False
        self.scan_end=0

        if fetch_config:
            #This will override the above properties IF the redis configuration hash is populated and exists
            self.configure_from_hash()
        elif self.input_json_dict is None:
            #We want to load the configuration to the redis hash
            config_dict={
                "hash_timeout":self.hash_timeout,
                "re_arm_time":self.re_arm_time,
                "fit_method":self.fit_method,
                "input_fixed_delays":self.input_fixed_delays,
                "input_fixed_phases":self.input_fixed_phases,
                "snr_threshold":self.snr_threshold
            }
            redis_publish_dict_to_hash(self.redis_obj, CONFIG_HASH, config_dict) 

        redis_clear_hash_contents(self.redis_obj, GPU_GAINS_REDIS_HASH)
        self.meta_obs = redis_hget_keyvalues(self.redis_obj, "META")
        self.ant_feng_map = ant_remotefeng_map.get_antennaFengineDict(self.redis_obj)
        if not self.dry_run:
            fixed_value_filepaths = redis_hget_keyvalues(self.redis_obj, CALIBRATION_CACHE_HASH)
            #Publish the initial fixed delays and trigger the F-Engines to load them
            self.log_and_post_slackmessage(f"""
            Publishing initial fixed-delays in
            `{fixed_value_filepaths["fixed_delay"]}`
            and fixed-phases in 
            `{fixed_value_filepaths["fixed_phase"]}`
            to F-Engines.""", severity="INFO")
            #fixed delays:
            load_delay_calibrations(fixed_value_filepaths["fixed_delay"])
            #fixed phases
            load_phase_calibrations(fixed_value_filepaths["fixed_phase"])
    
    def configure_from_hash(self):
        """This function will gather from the redis configuration hash, the required configuration in which
        to run the calibration process."""
        try:
            config = redis_hget_keyvalues(self.redis_obj,CONFIG_HASH)
        except:
            self.log_and_post_slackmessage(f"""
            Calibration process has been requested to run as a service,
            but the configuration redis hash {CONFIG_HASH},
            either does not exist or is inaccessible.
            Using default configuration.""")
            return
        
        self.hash_timeout = config.get("hash_timeout", self.hash_timeout)
        self.re_arm_time = config.get("re_arm_time", self.re_arm_time)
        self.fit_method = config.get("fit_method", self.fit_method)
        self.input_fixed_delays = config.get("input_fixed_delays", self.input_fixed_delays)
        self.input_fixed_phases = config.get("input_fixed_phases", self.input_fixed_phases)
        self.snr_threshold = config.get("snr_threshold", self.snr_threshold)

    def get_tuningidx_and_start_freq(self, message_key):
        key_split = message_key.split(',')
        tuning = key_split[-1]
        tuning_index = self.basebands.index(tuning)
        start_freq = float(key_split[0])*1e6

        return tuning_index, start_freq

    def update_antenna_phascals(self, ant_to_phasemap):
        redis_publish_dict_to_hash(self.redis_obj, "META_calibrationPhases",ant_to_phasemap)
        redis_publish_dict_to_channel(self.redis_obj, "update_calibration_phases", True)
    
    @staticmethod
    def dictnpy_to_dictlist(dictnpy):
        dictlst = {}
        for key in dictnpy:
            if np.iscomplexobj(dictnpy[key]):
                dictlst[key+"_real"] = dictnpy[key].real.tolist()
                dictlst[key+"_imag"] = dictnpy[key].imag.tolist()
            else:
                dictlst[key] = dictnpy[key].tolist()
        return dictlst

    def log_and_post_slackmessage(self, message, severity = "INFO", is_reply = False, update_message = False):
        try:
            level = getattr(logging, severity)
            logger.log(level, message)
            if level == 10: 
                return
        except: 
            logger.error(f"Invalid severity specified: {severity}.")

        if self.slackbot is not None:   
            if is_reply:
                self.slackbot.post_message(dedent(f"\
                    {severity}:\n" + message), thread_ts=self.slack_message_ts)
            elif update_message:
                self.slackbot.update_message(dedent(f"\
                    {severity}:\n" + message),
                    ts=self.slack_message_ts,
                )
            else:
                self.slackbot.post_message(dedent(f"\
                        {severity}:\n" + message))
                self.slack_message_ts = self.slackbot.last_message_ts

    def await_trigger(self):
        pubsub = self.redis_obj.pubsub(ignore_subscribe_messages=True)
        for channel in CHANNEL_ORDER:
            try:
                pubsub.subscribe(channel)
            except redis.RedisError:
                self.log_and_post_slackmessage(f'Subscription to "{channel}" unsuccessful.',severity = "ERROR",
                                            is_reply = True)
                return False
            
        self.log_and_post_slackmessage("Calibration process is armed and awaiting triggers from GPU nodes.",
                                       severity="INFO", is_reply = False)
        while True:
            #Fetch all messages
            while True:
                redis_publish_service_pulse(self.redis_obj, SERVICE_NAME)
                #Check if it is time to reset the calibration values:
                if time.time() >= self.scan_end and self.scan_is_ending:
                    self.log_and_post_slackmessage(f"""
                    Scan has ended at {time.ctime(self.scan_end)}, fixed delay and fixed phase values are being reset to
                    `{self.input_fixed_delays}`
                    and
                    `{self.input_fixed_phases}`
                    respectively.""", severity="INFO", is_reply = True)
                    load_delay_calibrations(self.input_fixed_delays)
                    load_phase_calibrations(self.input_fixed_phases)
                    self.scan_is_ending=False

                message = pubsub.get_message()
                if message:
                    #We have messages
                    if message['channel'] in CHANNEL_ORDER:
                        CHANNEL_MESSAGES[message['channel']].append(json.loads(message.get('data')))
                else:
                    if all(len(v) == 0 for v in CHANNEL_MESSAGES.values()):
                        continue
                    else:
                        #We've gotten some messages and no more messages left to fetch
                        break
            
            #Now process any of the messages received
            for channel in CHANNEL_ORDER:
                while len(CHANNEL_MESSAGES[channel])!=0:
                    #If the channel has a message i.e. list is not empty
                    message_data = CHANNEL_MESSAGES[channel].pop(0)
                    #Process message:
                    if channel == OBSERVATIONS_CHANNEL:
                        if "uvh5_calibrate" in message_data['postprocess']["#STAGES"]:
                            self.log_and_post_slackmessage(f"""
                            Received message indicating calibration observation is starting.
                            Collected mcast metadata now...""", severity = "INFO", is_reply = True)
                            self.projid = message_data['project_id']
                            self.dataset = message_data['dataset_id']
                            self.meta_obs = redis_hget_keyvalues(self.redis_obj, "META")
                        if "MoveARG" in message_data['postprocess']:
                            self.user_output_dir = (message_data['postprocess']["MoveARG"]).split('$',1)[0]
                            self.log_and_post_slackmessage(f"""
                            Upcoming observation is saving UVH5 files to:
                            `{self.user_output_dir}`
                            Calibration solutions and results will be saved to 
                            the same directory in folder `calibration`.
                            """, severity = "INFO", is_reply = True)
                            continue
                            
                    if channel == SCAN_END_CHANNEL:
                        self.scan_end = message_data["stop_time_unix"]
                        self.scan_is_ending=True
                        self.log_and_post_slackmessage(f"""
                        Calibration process has been notified that current scan with datasetID =
                        `{message_data['dataset_id']}`
                        is ending at {time.ctime(self.scan_end)}""", severity="INFO", is_reply=True)
                        continue

                    if channel == GPU_GAINS_REDIS_CHANNEL:
                        if message_data is not None:
                            self.log_and_post_slackmessage(f"""
                            GPU Gains message {message_data} received.""", severity="DEBUG")
                            return message_data
                        else:
                            continue                

    def collect_phases_for_hash_timeout(self, time_to_wait_until, manual_operation):
        """
        This function waits till time_to_wait expires, then collects GPU_calibrationGains,
        and processes hash contents.

        Args:
            time_to_wait_until : float, Unix time for the loop to wait until.
            manual_operation : boolean, dictates whether this is a manual run and self.input_json_dict is populated
                                        rather than fetching from Redis.
        Returns:
            - {<ant>_<tune_index> : [[complex(gains_pol0)], [complex(gains_pol1)]], ...}
            - collected frequency dict of {tune: [n_collected_freqs], ...}
            - list[antnames in observation]
            - obs_id of observation used for received gains
        """
        if manual_operation:
            calibration_gains = self.input_json_dict
        else:
            time.sleep(time_to_wait_until)
            calibration_gains = redis_hget_keyvalues(self.redis_obj, GPU_GAINS_REDIS_HASH)

        obs_id = None

        #Which antenna have been provided - assume unchanging given timeout expired
        ants = list(calibration_gains[list(calibration_gains.keys())[0]]['gains'].keys())

        #Initialise some empty dicts and arrays for population during this process
        collected_frequencies = {0:[],1:[]}
        ant_tune_to_collected_gain = {}
        for ant,tuning_idx in itertools.product(ants, range(self.nof_tunings)):
            ant_tune_to_collected_gain[ant+f"_{tuning_idx}"] = [[],[]]

        for start_freq_tune, payload in calibration_gains.items():
            try:
                tune_idx, start_freq = self.get_tuningidx_and_start_freq(start_freq_tune)
            except ValueError:
                self.log_and_post_slackmessage(f"UVH5 recorded baseband values not in {self.basebands}. Ignoring trigger.", severity="ERROR")
                raise ValueError
            self.log_and_post_slackmessage(f"Processing tuning {tune_idx}, start freq {start_freq}...", severity="DEBUG")
            obs_id_t = payload['obs_id']
            if obs_id is not None and obs_id_t != obs_id:
                self.log_and_post_slackmessage(f"""
                    Skipping {start_freq_tune} payload from GPU node since it contains differing obs_id 
                    {obs_id_t}
                    to previously encountered obs_id
                    {obs_id}.""", severity="WARNING", is_reply = True)
                continue
            else:
                obs_id = obs_id_t
            
            for ant, gain_dict in payload['gains'].items():
                key = ant+"_"+str(tune_idx)

                ant_tune_to_collected_gain[key][0] += [
                    complex(gain_dict["gain_pol0_real"][j], gain_dict["gain_pol0_imag"][j]) 
                    for j in range(len(gain_dict["gain_pol0_real"]))
                ]
                ant_tune_to_collected_gain[key][1] += [
                    complex(gain_dict["gain_pol1_real"][j], gain_dict["gain_pol1_imag"][j]) 
                    for j in range(len(gain_dict["gain_pol1_real"]))
                ]

            collected_frequencies[tune_idx] += payload['freqs_hz'] 
        
        return ant_tune_to_collected_gain, collected_frequencies, ants, obs_id

    def correctly_place_residual_phases_and_delays(self, ant_tune_to_collected_gain, 
        collected_frequencies, full_observation_channel_frequencies):
        """
        First, `collected_frequencies` need to be ordered as they are not ordered when collected. Their ordering
        will be the same as what is needed for the gains (since they are taken 1 to 1).
        After ordering the `collected_frequencies` -> either ascending or descending, the indices of their placement
        inside of the `full_observation_channel_frequencies_hz` must be calculated. 
        By investigating the placement of `collected_frequencies` inside of `full_observation_channel_frequencies_hz`,
        a map of how to place the collected gains inside an array of `self.nof_channels` per stream per antenna
        may be generated.

        Args:
            ant_tune_to_collected_gain : {<ant>_<tune_index> : [[complex(gains_pol0)], [complex(gains_pol1)]], ...}
            collected_frequences: collected frequency dict of {tune: [n_collected_freqs], ...}
            full_observation_channel_frequencies_hz: a matrix of dims(nof_tunings, nof_channels)

        Returns:
            ant_gains_map : a dictionary mapping of {ant: [nof_streams, nof_frequencies]}
            frequency_indices : a dictionary mapping of {tuning_idx : np.array(int)}
        """
        full_gains_map = {}
        sortings = {}
        frequency_indices = {}

        #Generate our sortings and sort frequencies. Also find placement of sorted collected
        #frequencies in the full nof_chan frequencies.
        for tuning, collected_freq in collected_frequencies.items():
            collected_freq = np.array(collected_freq,dtype=float)
            #Get the indices that would sort the frequencies into ascending order.
            sort_indices = np.argsort(collected_freq)
            #Find out in which direction the frequencies run
            obs_frequencies_ascending = np.all(np.diff(full_observation_channel_frequencies[tuning,:]) > 0)
            if not obs_frequencies_ascending:
                sort_indices = sort_indices[::-1]
            #sort frequencies - either into ascending or descending order
            collected_frequencies[tuning] = np.round(collected_freq[sort_indices],decimals=2)
            #find sorted frequency indices inside of overall observation frequencies
            frequency_indices[tuning] = (np.searchsorted(full_observation_channel_frequencies[tuning,:],collected_frequencies[tuning])
                                        if obs_frequencies_ascending
                                        else
                                        full_observation_channel_frequencies[tuning,:].size-1-np.searchsorted(full_observation_channel_frequencies[tuning,::-1],collected_frequencies[tuning]))

            if not np.all(np.isin(collected_frequencies[tuning], full_observation_channel_frequencies[tuning,:])):
                self.log_and_post_slackmessage(f"""
                Not all collected frequencies are present inside those calculated to be the expected observation frequencies for tuning {tuning}.
                Collected frequencies span range:
                {collected_frequencies[tuning][0]} -> {collected_frequencies[tuning][-1]}Hz
                Aborting run.""", severity="ERROR", is_reply=True)
                return None, None

            #store the sortings for sorting the gains later
            sortings[tuning] = sort_indices
        
        nof_chans_collected_0 = len(collected_frequencies[0]) 
        nof_chans_collected_1 = len(collected_frequencies[1]) 
        percent_collected_0 = (nof_chans_collected_0 / self.nof_channels) * 100.0
        percent_collected_1 = (nof_chans_collected_1 / self.nof_channels) * 100.0
        log_message = """-------------------------------------------------------------"""
        try:
            log_message += f"""\n
            Calculating phase and delay residuals for tuning 0 off of:
            {nof_chans_collected_0} collected gains out of {self.nof_channels} = {percent_collected_0}%.
            Total observation frequencies for tuning 0 are expected to span: 
            `{full_observation_channel_frequencies[0,0]}->{full_observation_channel_frequencies[0,-1]}Hz`"""
        except IndexError:
            log_message += f"""\n
            No values received for tuning 0, and so no residuals will be calculated for that tuning."""
        try:
             log_message += f"""\n
            Calculating phase and delay residuals for tuning 1 off of:
            {nof_chans_collected_1} collected gains out of {self.nof_channels} = {percent_collected_1}%.
            Total observation frequencies for tuning 1 are expected to span: 
            `{full_observation_channel_frequencies[1,0]}->{full_observation_channel_frequencies[1,-1]}Hz`"""
        except IndexError:
            log_message += f"""\n
            No values received for tuning 1, and so no residuals will be calculated for that tuning."""
        log_message += "\n-------------------------------------------------------------"
        self.log_and_post_slackmessage(log_message, severity="INFO", is_reply=True)

        #Initialise full ant_gains_map
        gain_zeros = np.zeros((self.nof_streams, self.nof_channels),dtype=np.complex64)
        for ant in self.ants:
            full_gains_map[ant] = gain_zeros.copy()

        #sort gains according to sorting of frequencies, and place them in nof_chan array correctly
        for ant_tune, gains in ant_tune_to_collected_gain.items():
            gains = np.array(gains,dtype=np.complex64)
            ant, tune = ant_tune.split('_')
            tune = int(tune)
            #per antenna, per tuning
            sorted_gains = gains[:,sortings[tune]]
            try:
                full_gains_map[ant][tune*2, frequency_indices[tune]] = sorted_gains[0]
                full_gains_map[ant][(tune*2)+1, frequency_indices[tune]] = sorted_gains[1]
            except:
                try:
                    self.log_and_post_slackmessage(f"""
                    Could not place received frequencies within expected observation frequencies.
                    This likely occured due to the collected fcent not matching fcent used in recording.

                    Recorded gains have frequencies for tuning 0:
                    `{collected_frequencies[0][0]}->{collected_frequencies[0][-1]}Hz`
                    and tuning 1:
                    `{collected_frequencies[1][0]}->{collected_frequencies[1][-1]}Hz`

                    Ignoring run.
                    """,severity = "ERROR", is_reply = True)
                except:
                    self.log_and_post_slackmessage(f"""
                    Could not place received frequencies within expected observation frequencies.
                    This likely occured due to the collected fcent not matching fcent used in recording.

                    Ignoring run.
                    """,severity = "ERROR", is_reply = True)
                return None, None

        return full_gains_map, frequency_indices

    def start(self):
        while True:
            #clear upfront incase bad calibration gains cause continuation of loop
            redis_clear_hash_contents(self.redis_obj, GPU_GAINS_REDIS_HASH)
            manual_operation = False
            #Launch function that waits for first valid message:
            if self.input_json_dict is None:
                trigger = self.await_trigger()
            else:
                #in this case, the operation is running manually with input json files
                manual_operation = True
                trigger = True
            if trigger:
                self.configure_from_hash()
                self.log_and_post_slackmessage(f"""
                    Calibration process has been triggered and is starting.\n
                    Manual run = `{manual_operation}`, 
                    Dry run = `{self.dry_run}`,
                    hash timeout = `{self.hash_timeout}s`, re-arm time = `{self.re_arm_time}s`,
                    fitting method = `{self.fit_method}`,
                    snr threshold = `{self.snr_threshold}`,
                    output directory = `{self.user_output_dir}`,
                    projid = {self.projid},
                    dataset_id = {self.dataset}""", severity = "INFO",
                    is_reply=False, update_message=True)

                #FOR SPOOFING - TEMPORARY AND NEEDS TO BE SMARTER:
                self.basebands = [
                    "AC",
                    "BD"
                    ]

                #Start function that waits for hash_timeout before collecting redis hash.
                try:
                    ant_tune_to_collected_gains, collected_frequencies, self.ants, obs_id = self.collect_phases_for_hash_timeout(self.hash_timeout, manual_operation = manual_operation) 
                except Exception as e:
                    self.log_and_post_slackmessage(f"""
                    The collection of calibration from GPU gains failed:
                    {e}
                    Ignoring current trigger.""", severity="ERROR")
                    if manual_operation:
                        return
                    continue
                
                #Update output_dir to contain projid, dataset_id and obs_id
                output_dir = self.user_output_dir if manual_operation else os.path.join(self.user_output_dir, self.projid, self.dataset, obs_id, "calibration")

                if manual_operation:
                    fcents_mhz = np.array([float(fcent) for fcent in self.fcents],dtype=float)
                    sideband = np.array([float(sb) for sb in self.input_sideband],dtype=float)
                    tbin = float(self.tbin)
                else:
                    fcents_mhz = np.array(self.meta_obs["fcents"],dtype=float)
                    tbin = self.meta_obs["tbin"]
                    sideband = self.meta_obs["sideband"]
                    
                channel_bw = 1/tbin
                fcent_hz = fcents_mhz*1e6
                self.source = self.meta_obs["src"]

                self.log_and_post_slackmessage(f"""
                    Observation meta reports:
                    `source = {self.source}`
                    `basebands = {self.basebands}`
                    `sidebands = {sideband}`
                    `fcents = {fcents_mhz} MHz`
                    `tbin = {tbin}`""",
                    severity = "INFO", is_reply=True)

                full_observation_channel_frequencies_hz = np.round(np.vstack((
                    fcent_hz[0] + np.arange(-self.nof_channels//2, self.nof_channels//2) * channel_bw * sideband[0],
                    fcent_hz[1] + np.arange(-self.nof_channels//2, self.nof_channels//2) * channel_bw * sideband[1]
                )),decimals=2)

                full_gains_map, frequency_indices = self.correctly_place_residual_phases_and_delays(
                    ant_tune_to_collected_gains, collected_frequencies, 
                    full_observation_channel_frequencies_hz
                )
                if full_gains_map is None:
                    if manual_operation:
                        return
                    continue
                
                #-------------------------SAVE COLLECTED GAINS-------------------------#
                collected_gain_path = os.path.join(output_dir,f"gains_{obs_id}.json")
                #For json dumping:
                try:
                    t_full_gains_map = self.dictnpy_to_dictlist(full_gains_map)

                    with open(collected_gain_path, 'w') as f:
                        json.dump(t_full_gains_map, f)

                    self.log_and_post_slackmessage(f"""
                    Saving full collected gains dictionary mapping to:
                    `{collected_gain_path}`
                    """,severity="INFO",is_reply = True)
                except Exception as e:
                    self.log_and_post_slackmessage(f"""
                    Unable to save calibration gains dictionary. Continuing...
                    """,severity="WARNING",is_reply = True)
                    self.log_and_post_slackmessage(f"""
                    Exception:
                    {e}""", severity="DEBUG")
                    
                #-------------------------PLOT PHASE OF COLLECTED GAINS-------------------------#
                self.log_and_post_slackmessage("""
                Plotting phase and amplitude of the collected recorded gains...
                """,severity="DEBUG")

                phase_file_path_ac, phase_file_path_bd = plot_gain_phase(full_gains_map, full_observation_channel_frequencies_hz, frequency_indices, 
                                                                        fit_method = self.fit_method,
                                                                        outdir = os.path.join(output_dir, "calibration_plots"), outfilestem=obs_id,
                                                                        source_name = self.source)
                amplitude_file_path_ac, amplitude_file_path_bd = plot_gain_amplitude(full_gains_map, full_observation_channel_frequencies_hz, frequency_indices,
                                                                        outdir = os.path.join(output_dir, "calibration_plots"), outfilestem=obs_id,
                                                                        source_name = self.source)

                self.log_and_post_slackmessage(f"""
                        Saved recorded gain phase for tuning AC to: 
                        `{phase_file_path_ac}`
                        and BD to:
                        `{phase_file_path_bd}`
                        """, severity = "DEBUG")
                self.log_and_post_slackmessage(f"""
                        Saved recorded gain amplitude for tuning AC to: 
                        `{amplitude_file_path_ac}`
                        and BD to:
                        `{amplitude_file_path_bd}`
                        """, severity = "DEBUG")
                
                if self.slackbot is not None:
                    try:
                        self.slackbot.upload_file(phase_file_path_ac, title =f"Recorded phases (degrees) for tuning AC from\n`{obs_id}`",
                                                thread_ts = self.slack_message_ts)
                        self.slackbot.upload_file(phase_file_path_bd, title =f"Recorded phases (degrees) for tuning BD from\n`{obs_id}`",
                                                thread_ts = self.slack_message_ts)
                        self.slackbot.upload_file(amplitude_file_path_ac, title =f"Recorded amplitude for tuning AC from\n`{obs_id}`",
                                                thread_ts = self.slack_message_ts)
                        self.slackbot.upload_file(amplitude_file_path_bd, title =f"Recorded amplitude for tuning BD from\n`{obs_id}`",
                                                thread_ts = self.slack_message_ts)
                    except:
                        self.log_and_post_slackmessage("Error uploading plots", severity="WARNING", is_reply=True)

                if not manual_operation:
                    fixed_phase_filepath = redis_hget_keyvalues(self.redis_obj, CALIBRATION_CACHE_HASH)["fixed_phase"]
                else:
                    fixed_phase_filepath = self.input_fixed_phases
                try:
                    with open(fixed_phase_filepath, 'r') as f:
                        last_fixed_phases = json.load(f)
                except:
                    self.log_and_post_slackmessage(f"""
                        Could *not* read fixed phases from {fixed_phase_filepath} for updating with calculated residuals.
                        Cleaning up and aborting calibration process...
                    """, severity = "ERROR", is_reply= True)
                    return

                ant_to_grade = calc_calibration_ant_grade(full_gains_map)
                freq_to_grade = calc_calibration_freq_grade(full_gains_map)
                full_grade = calc_full_grade(full_gains_map)
                grade_file_path = plot_gain_grade(ant_to_grade, freq_to_grade, outdir=os.path.join(output_dir ,"calibration_plots"), outfilestem=obs_id,
                        source_name = self.source)
                self.log_and_post_slackmessage(f"""
                        Saved calibration gain grade plot to: 
                        `{grade_file_path}`
                        """, severity = "DEBUG")
                if self.slackbot is not None:
                    try:
                        self.slackbot.upload_file(grade_file_path,
                                                title =f"Calibration gain grade from\n`{obs_id}`",
                                                thread_ts = self.slack_message_ts)
                    except:
                        self.log_and_post_slackmessage("Error uploading plots", severity="WARNING", is_reply=True)
                self.log_and_post_slackmessage(f"""
                        Calculated overall grade for calibration recording of:
                        `{full_grade}`
                        """, severity = "INFO", is_reply=True)

                self.log_and_post_slackmessage(f"""
                Subtracting fixed phases found in
                ```{fixed_phase_filepath}```
                from the received gain matrix
                """, severity = "INFO", is_reply=True)

                #-------------------------CALCULATE RESIDUAL DELAYS AND PHASES FOR COLLECTED GAINS-------------------------#
                try:
                    if self.fit_method == "linear":
                        delay_residual_map, phase_cal_map = calc_residuals_from_polyfit(full_gains_map, full_observation_channel_frequencies_hz,
                                                                                        last_fixed_phases, frequency_indices, snr_threshold = self.snr_threshold)
                    elif self.fit_method == "fourier":
                        delay_residual_map, phase_cal_map, snr_map, sigma_phase_map = calc_residuals_from_ifft(full_gains_map,full_observation_channel_frequencies_hz,
                                                                                    last_fixed_phases, frequency_indices, sideband, snr_threshold = self.snr_threshold)
                except Exception as e:
                    self.log_and_post_slackmessage(f"""
                    Exception encountered making a call to the calibration kernel:
                    {e}
                    Ignoring run and continuing...
                    """, severity = "ERROR", is_reply=True)
                    if manual_operation:
                        return
                    continue

                #-------------------------SAVE RESIDUAL DELAYS-------------------------#
                #For json dumping:
                t_delay_dict = self.dictnpy_to_dictlist(delay_residual_map)

                #log directory for calibration delay residuals
                delay_residual_path = os.path.join(output_dir, "delay_residuals")
                if not os.path.exists(delay_residual_path):
                    os.makedirs(delay_residual_path)

                delay_residual_filename = os.path.join(delay_residual_path, f"calibrationdelayresiduals_{obs_id}.json")
                with open(delay_residual_filename, 'w') as f:
                    json.dump(t_delay_dict, f)
                self.log_and_post_slackmessage(f"""
                    Wrote out calculated *residual delays* to: 
                    {delay_residual_filename}""", severity = "DEBUG")
                redis_publish_dict_to_hash(self.redis_obj, "META_residualDelays", t_delay_dict)

                pretty_print_json = pprint.pformat(json.dumps(t_delay_dict)).replace("'", '"')
                self.log_and_post_slackmessage(f"""
                    Calculated the following delay residuals from UVH5 recording
                    `{obs_id}`:

                    ```{pretty_print_json}```
                    """, severity = "INFO", is_reply=True)

                #-------------------------UPDATE THE FIXED DELAYS-------------------------#
                if not manual_operation:
                    fixed_delay_filepath = redis_hget_keyvalues(self.redis_obj, CALIBRATION_CACHE_HASH)["fixed_delay"]
                else:
                    fixed_delay_filepath = self.input_fixed_delays
                try:
                    fixed_delays = pd.read_csv(os.path.abspath(fixed_delay_filepath), names = ["IF0","IF1","IF2","IF3"],
                            header=None, skiprows=1)
                except:
                    self.log_and_post_slackmessage(f"""
                        Could *not* read fixed delays from {fixed_delay_filepath} for updating with calculated residuals.
                        Clearning up and aborting calibration process...
                    """, severity = "ERROR", is_reply=True)
                    return
                self.log_and_post_slackmessage(f"""
                Modifying fixed-delays found in
                ```{fixed_delay_filepath}```
                with the *residual delays* calculated above in
                ```{delay_residual_filename}```
                """,severity="INFO", is_reply=True)
                fixed_delays = fixed_delays.to_dict()
                updated_fixed_delays = {}
                for i, tune in enumerate(list(fixed_delays.keys())):
                    sub_updated_fixed_delays = {}
                    for ant, delay in fixed_delays[tune].items():
                        if ant in delay_residual_map:
                            sub_updated_fixed_delays[ant] = delay - delay_residual_map[ant][i]
                        else:
                            sub_updated_fixed_delays[ant] = delay
                    updated_fixed_delays[tune] = sub_updated_fixed_delays

                #bit of logic here to remove the previous filestem from the name.
                fixed_delay_file_loc = os.path.join((output_dir), ("fixed_delays/"))
                if not os.path.exists(fixed_delay_file_loc):
                    os.makedirs(fixed_delay_file_loc)
                if '%' in fixed_delay_filepath:
                    modified_fixed_delays_path = os.path.join(fixed_delay_file_loc, (os.path.splitext(os.path.basename(fixed_delay_filepath))[0].split('%')[1]+"%"+obs_id+".csv"))
                #if first time running
                else:
                    modified_fixed_delays_path = os.path.join(fixed_delay_file_loc, (os.path.splitext(os.path.basename(fixed_delay_filepath))[0]+"%"+obs_id+".csv"))
                
                self.log_and_post_slackmessage(f"""
                    Wrote out modified fixed delays to: 
                    ```{modified_fixed_delays_path}```
                    """, severity = "INFO", is_reply=True)

                df = pd.DataFrame.from_dict(updated_fixed_delays)
                df.to_csv(modified_fixed_delays_path)

                #Publish new fixed delays to FEngines:
                if not self.dry_run:
                    self.log_and_post_slackmessage("""Updating fixed-delays on *all* antenna now...""", severity = "INFO", is_reply=True)
                    load_delay_calibrations(modified_fixed_delays_path)
                
                #-------------------------LOAD THE NEW FIXED PHASES-------------------------#

                #bit of logic here to remove the previous filestem from the name.
                fixed_phase_file_loc = os.path.join((output_dir), ("fixed_phases/"))
                if not os.path.exists(fixed_phase_file_loc):
                    os.makedirs(fixed_phase_file_loc)
                if '%' in fixed_phase_filepath:
                    modified_fixed_phases_path = os.path.join(fixed_phase_file_loc, (os.path.splitext(os.path.basename(fixed_phase_filepath))[0].split('%')[1]+"%"+obs_id+".json"))   
                #if first time running
                else:
                    modified_fixed_phases_path = os.path.join(fixed_phase_file_loc, (os.path.splitext(os.path.basename(fixed_phase_filepath))[0]+"%"+obs_id+".json"))

                self.log_and_post_slackmessage(f"""
                Wrote out modified fixed phases to: 
                ```{modified_fixed_phases_path}```""", severity = "INFO", is_reply=True)

                t_phase_cal_map = self.dictnpy_to_dictlist(phase_cal_map)
                with open(modified_fixed_phases_path, 'w+') as f:
                    json.dump(t_phase_cal_map, f)

                # Update the fixed phases on the F-Engines and update the fixed_phase path
                if not self.dry_run:
                    self.log_and_post_slackmessage("""Updating fixed-phases on *all* antenna now...""", severity = "INFO", is_reply=True)
                    load_phase_calibrations(modified_fixed_phases_path)
                    
                #-------------------------PLOT GENERATION AND SAVING-------------------------#
                #Plot phase and delay residuals
                delay_file_path, phase_file_path_ac, phase_file_path_bd = plot_delay_phase(delay_residual_map, phase_cal_map, 
                        full_observation_channel_frequencies_hz, outdir = os.path.join(output_dir ,"calibration_plots"), outfilestem=obs_id,
                        source_name = self.source)

                self.log_and_post_slackmessage(f"""
                        Saved  residual delay plot to: 
                        `{delay_file_path}`
                        and phase plot to:
                        `{phase_file_path_ac}`
                        and
                        `{phase_file_path_bd}`
                        """, severity = "DEBUG")

                if self.slackbot is not None:
                    try:
                        self.slackbot.upload_file(delay_file_path,
                                                title =f"Residual delays (ns) per antenna calculated from\n`{obs_id}`",
                                                thread_ts = self.slack_message_ts)
                        self.slackbot.upload_file(phase_file_path_ac,
                                                title =f"Phases (degrees) per frequency (Hz) for tuning AC calculated from\n`{obs_id}`",
                                                thread_ts = self.slack_message_ts)
                        self.slackbot.upload_file(phase_file_path_bd,
                                                title =f"Phases (degrees) per frequency (Hz) for tuning BD calculated from\n`{obs_id}`",
                                                thread_ts = self.slack_message_ts)
                    except:
                        self.log_and_post_slackmessage("Error uploading plots", severity="WARNING", is_reply=True)
                
                #Plot SNR of delay peak and std deviation of phases
                snr_and_sigma_file_path = plot_snr_and_phase_spread(snr_map, sigma_phase_map, outdir = os.path.join(output_dir ,"calibration_plots"), outfilestem=obs_id,
                        source_name = self.source)
                
                self.log_and_post_slackmessage(f"""
                        Saved  snr and phase spread plot to: 
                        `{snr_and_sigma_file_path}`
                        """, severity = "DEBUG")
                if self.slackbot is not None:
                    try:
                        self.slackbot.upload_file(snr_and_sigma_file_path,
                                                title =f"Delay peak SNR and std_deviation of phases from\n`{obs_id}`",
                                                thread_ts = self.slack_message_ts)
                    except:
                        self.log_and_post_slackmessage("Error uploading plots", severity="WARNING", is_reply=True)

                #-------------------------FINISH OFF CALIBRATION RUN-------------------------#
                if manual_operation:
                    self.log_and_post_slackmessage(f"""
                    Manual calibration process run complete.
                    """)
                    return
                else:
                    #Ensure permissions are correct on the calibration output folder
                    try:
                        os.system(f"chown cosmic:cosmic -R {output_dir}")
                    except:
                        pass
                    #Sleep
                    self.log_and_post_slackmessage(f"""
                        Sleeping for {self.re_arm_time}s. 
                        Will not detect any channel triggers during this time.
                        """, severity = "INFO",is_reply=True)
                    time.sleep(self.re_arm_time)
                    self.log_and_post_slackmessage(f"""
                        Calibration run for
                        `{obs_id}`
                        is complete!
                        Test run = `{manual_operation}`, 
                        Dry run = `{self.dry_run}`,
                        hash timeout = `{self.hash_timeout}s`, re-arm time = `{self.re_arm_time}s`,
                        fitting method = `{self.fit_method}`,
                        snr threshold = `{self.snr_threshold}`,
                        source = `{self.source}`, 
                        results directory = `{output_dir}`
                    """, severity="INFO", is_reply=False, update_message=True)
                    self.log_and_post_slackmessage(f"""
                        Clearing redis hash: {GPU_GAINS_REDIS_HASH} contents in anticipation of next calibration run.
                        """,severity = "DEBUG")
            else:
                self.log_and_post_slackmessage(f"""
                Issue waiting on trigger from GPU nodes. Aborting calibration proces...
                """, severity = "ERROR", is_reply = False, update_message = True)
                exit(0)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
    description=("""Listen for updates to GPU hashes containing calibration phases
    and generate residual delays and load calibration phases to the F-Engines.""")
    )
    parser.add_argument("-s","--run-as-service", action="store_true",help="""If specified, all other arguments are ignored
    and the configuration set up is collected on each main loop from the configuration redis Hash. See 
    configure_calibration_process.py to set the hash contents for configuration.""")
    parser.add_argument("--hash-timeout", type=float,default=10, required=False, help="""How long to wait for calibration 
    postprocessing to complete and update phases.""")
    parser.add_argument("--dry-run", action="store_true", help="""If run as a dry run, delay residuals and phases are 
    calcualted and written to redis/file but not loaded to the F-Engines nor applied to the existing fixed-delays.""")
    parser.add_argument("--re-arm-time", type=float, default=20, required=False, help="""After collecting phases
    from GPU nodes and performing necessary actions, the service will sleep for this duration until re-arming""")
    parser.add_argument("--fit-method", type=str, default="fourier", required=False, help="""Pick the complex fitting method
    to use for residual calculation. Options are: ["linear", "fourier"]""")
    parser.add_argument("-o", "--output-dir", type=str, default="/mnt/cosmic-storage-2/data2", required=False, help="""The output directory in 
    which to place all log folders/files during operation.""")
    parser.add_argument("-f","--fixed-delay-to-update", type=str, required=False, help="""
    csv file path to latest fixed delays that must be modified by the residual delays calculated in this script. If not provided,
    process will try use fixed-delay file path in cache.""")
    parser.add_argument("-p","--fixed-phase-to-update", type=str, required=False, help="""
    json file path to latest fixed phases that must be modified by the residual phases calculated in this script. If not provided,
    process will try use fixed-phase file path in cache.""")
    parser.add_argument("--no-slack-post", action="store_true",help="""If specified, logs are not posted to slack.""")
    parser.add_argument("--fcentmhz", nargs="*", default=[1000, 1001], help="""fcent values separated by space for observation""")
    parser.add_argument("--tbin", type=float, default=1e-6, required=False, help="""tbin value for observation in seconds""")
    parser.add_argument("--sideband", nargs="*", default=[1, 1], help="sideband values separated by space for observation")
    parser.add_argument("--snr-threshold", type=float, default = 4.0, required=False, 
                        help="""The snr threshold above which the process will reject applying the calculated delay
                        and phase residual calibration values""")
    parser.add_argument('file', type=argparse.FileType('r'), nargs='*')
    args = parser.parse_args()
    manual_run = False
    input_json_dict = {}
    if len(args.file) != 0:
        for f in args.file:
            input_json_dict.update(json.load(f))
        args.dry_run = True
        manual_run = True

    #Logs contain traceback exceptions
    def exception_hook(*args):
        logger.error("".join(traceback.format_exception(*args)))
        print("".join(traceback.format_exception(*args)))
    sys.excepthook = exception_hook

    def _exit():
        # this happens after exception_hook even in the event of an exception
        logger.info("Exiting.")
        print("Exiting.")
    atexit.register(_exit)
    
    slackbot = None
    if not args.no_slack_post:
        if "SLACK_BOT_TOKEN" in os.environ:
            slackbot = SlackBot(os.environ["SLACK_BOT_TOKEN"], chan_name="active_vla_calibrations", chan_id="C04KTCX4MNV")
            topic = f"*Logging the VLA calibration in the loop*"
            slackbot.set_topic(topic)
            slackbot.post_message(f"""
            Starting calibration observation process...""")
        else:
            logger.log(getattr(logging, "INFO"), "SLACK_BOT_TOKEN not in environment keys. Will not log to slack.")
        
    #if input fixed delay and fixed phase files are provided, publish them to the filepath hash
    input_fixed_delays = args.fixed_delay_to_update    
    if input_fixed_delays is not None:
        if not manual_run:
            redis_publish_dict_to_hash(redis_obj, CALIBRATION_CACHE_HASH,{"fixed_delay":input_fixed_delays})
            input_fixed_delays = None
    input_fixed_phases = args.fixed_phase_to_update
    if input_fixed_phases is not None:
        if not  manual_run:
            redis_publish_dict_to_hash(redis_obj, CALIBRATION_CACHE_HASH,{"fixed_phase":input_fixed_phases})
            input_fixed_phases = None
    
    output_dir = os.path.abspath(args.output_dir)
    if output_dir is not None and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    calibrationGainCollector = CalibrationGainCollector(redis_obj, fetch_config = args.run_as_service, user_output_dir = output_dir, hash_timeout = args.hash_timeout, dry_run = args.dry_run,
                                re_arm_time = args.re_arm_time, fit_method = args.fit_method, slackbot = slackbot, input_fixed_delays = input_fixed_delays,
                                input_fixed_phases = input_fixed_phases, input_json_dict = None if not bool(input_json_dict) else input_json_dict,
                                input_fcents = args.fcentmhz, input_sideband = args.sideband, input_tbin = args.tbin, snr_threshold = args.snr_threshold)
    calibrationGainCollector.start()
