import pandas as pd
import numpy as np
import itertools
import argparse
import os
import logging
from logging.handlers import RotatingFileHandler
import redis
import time
import json
from delaycalibration import DelayCalibrationWriter
from textwrap import dedent
import pprint
from calibration_residual_kernals import calc_residuals_from_polyfit, calc_residuals_from_ifft
from cosmic.observations.slackbot import SlackBot
from cosmic.fengines import ant_remotefeng_map
from cosmic.redis_actions import redis_obj, redis_hget_keyvalues, redis_publish_dict_to_hash, redis_clear_hash_contents, redis_publish_service_pulse, redis_publish_dict_to_channel
from plot_delay_phase import plot_delay_phase, plot_gain_phase

LOGFILENAME = "/home/cosmic/logs/DelayCalibration.log"
logger = logging.getLogger('calibration_delays')
logger.setLevel(logging.DEBUG)

SERVICE_NAME = os.path.splitext(os.path.basename(__file__))[0]

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

GPU_PHASES_REDIS_HASH = "GPU_calibrationPhases"
GPU_GAINS_REDIS_HASH = "GPU_calibrationGains"
GPU_PHASES_REDIS_CHANNEL = "gpu_calibrationphases"
GPU_GAINS_REDIS_CHANNEL = "gpu_calibrationgains"

CALIBRATION_CACHE_HASH = "CAL_fixedValuePaths"

class CalibrationGainCollector():
    def __init__(self, redis_obj, output_dir, hash_timeout=20, re_arm_time = 30, fit_method = "linear", dry_run = False,
    nof_streams = 4, nof_tunings = 2, nof_pols = 2, nof_channels = 1024, slackbot=None, input_json_dict = None, input_fcents = None, input_tbin = None):
        self.redis_obj = redis_obj
        self.output_dir = output_dir
        self.hash_timeout = hash_timeout
        self.re_arm_time = re_arm_time
        self.fit_method = fit_method
        self.dry_run = dry_run
        self.slackbot = slackbot
        self.input_json_dict = input_json_dict
        self.fcents = input_fcents
        self.tbin = input_tbin
        self.nof_streams = nof_streams
        self.nof_channels = nof_channels
        self.nof_tunings = nof_tunings
        self.nof_pols = nof_pols
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
            self.delay_calibration = DelayCalibrationWriter(self.redis_obj, fixed_value_filepaths["fixed_delay"])
            self.delay_calibration.run()
            #fixed phases
            with open(fixed_value_filepaths["fixed_phase"], 'r') as f:
                self.update_antenna_phascals(json.load(f))
    
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
            dictlst[key] = dictnpy[key].tolist()
        return dictlst

    def log_and_post_slackmessage(self, message, severity = "INFO"):
        if severity =="INFO":
            logger.info(message)
            if self.slackbot is not None:
                self.slackbot.post_message(dedent("\
                    INFO:\n" + message))
        if severity =="WARN":
            logger.warn(message)
            if self.slackbot is not None:
                self.slackbot.post_message(dedent("\
                    WARNING:\n" + message))
        if severity =="ERROR":
            logger.error(message)
            if self.slackbot is not None:
                self.slackbot.post_message(dedent("\
                    ERROR:\n" + message))
        if severity == "DEBUG":
            logger.debug(message)

    def await_trigger(self):
        pubsub = self.redis_obj.pubsub(ignore_subscribe_messages=True)
        try:
            pubsub.subscribe(GPU_GAINS_REDIS_CHANNEL)
        except redis.RedisError:
            self.log_and_post_slackmessage(f"""
                Unable to subscribe to {GPU_GAINS_REDIS_CHANNEL} 
                channel to listen for changes to {GPU_GAINS_REDIS_HASH}.""",
            severity="ERROR")
            return False
        try:
            pubsub.subscribe("observations")
        except:
            self.log_and_post_slackmessage(f'Subscription to "observations" unsuccessful.',severity = "ERROR")

        self.log_and_post_slackmessage("Calibration process is armed and awaiting triggers from GPU nodes.", severity="INFO")

        for message in pubsub.listen():
            redis_publish_service_pulse(self.redis_obj, SERVICE_NAME)
            if message is not None and isinstance(message, dict):
                msg = json.loads(message.get('data'))
                if message['channel'] == "observations":
                    if msg['postprocess'] == "calibrate-uvh5":
                        self.log_and_post_slackmessage(f"""
                        Received message indicating calibration observation is starting.
                        Collected mcast metadata now...""", severity = "INFO")
                        self.meta_obs = redis_hget_keyvalues(self.redis_obj, "META")
                        continue
                if message['channel'] == GPU_GAINS_REDIS_CHANNEL:
                    if msg is not None:
                        return msg
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
            tune_idx, start_freq = self.get_tuningidx_and_start_freq(start_freq_tune)
            self.log_and_post_slackmessage(f"Processing tuning {tune_idx}, start freq {start_freq}...", severity="DEBUG")
            obs_id_t = payload['obs_id']
            if obs_id is not None and obs_id_t != obs_id:
                self.log_and_post_slackmessage(f"""
                    Skipping {start_freq_tune} payload since it contains differing obs_id {obs_id_t}
                    to previously encountered obs_id {obs_id}.""", severity="WARNING")
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
        By investigating the placement of `collected_frequencies` inside of `full_observation_channel_frequencies_hz`,
        a map of how to place the collected gains inside an array of `self.nof_channels` per stream per antenna
        may be generated.

        Args:
            ant_tune_to_collected_gain : {<ant>_<tune_index> : [[complex(gains_pol0)], [complex(gains_pol1)]], ...}
            collected_frequences: collected frequency dict of {tune: [n_collected_freqs], ...}
            full_observation_channel_frequencies_hz: a matrix of dims(nof_tunings, nof_channels)

        Returns:
            ant_gains_map : a dictionary mapping of {ant: [nof_streams, nof_frequencies]}
            chan_start : integer index for the start frequency of those collected out of all observation channel frequencies
            chan_stop : integer index for the stop frequency of those collected out of all observation channel frequencies
        """
        full_gains_map = {}
        sortings = {}
        frequency_indices = {}

        #Generate our sortings and sort frequencies. Also find placement of sorted collected
        #frequencies in the full nof_chan frequencies.
        for tuning, collected_freq in collected_frequencies.items():
            collected_freq = np.array(collected_freq,dtype=float)
            #find sorting
            sortings[tuning] = np.argsort(collected_freq)
            #sort frequencies
            collected_frequencies[tuning] = collected_freq[sortings[tuning]]
            #find sorted frequency indices
            frequency_indices[tuning] = full_observation_channel_frequencies[tuning,:].searchsorted(collected_frequencies[tuning])
        
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
        self.log_and_post_slackmessage(log_message, severity="INFO")

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

            full_gains_map[ant][tune*2, frequency_indices[tune]] = sorted_gains[0]
            full_gains_map[ant][(tune*2)+1, frequency_indices[tune]] = sorted_gains[1]

        return full_gains_map, frequency_indices

    def start(self):
        while True:
            manual_operation = False
            #Launch function that waits for first valid message:
            if self.input_json_dict is None:
                trigger = self.await_trigger()
            else:
                #in this case, the operation is running manually with input json files
                manual_operation = True
                trigger = True
            if trigger:
                self.log_and_post_slackmessage(f"""
                    Calibration process has been triggered.\n
                    Manual run = `{manual_operation}`, 
                    Dry run = `{self.dry_run}`,
                    hash timeout = `{self.hash_timeout}`s, re-arm time = `{self.re_arm_time}`s,
                    fitting method = `{self.fit_method}`,
                    output directory = `{self.output_dir}`""", severity = "INFO")

                #FOR SPOOFING - TEMPORARY AND NEEDS TO BE SMARTER:
                self.basebands = [
                    "AC_8BIT",
                    "BD_8BIT"
                    ]

                #Start function that waits for hash_timeout before collecting redis hash.
                ant_tune_to_collected_gains, collected_frequencies, self.ants, obs_id = self.collect_phases_for_hash_timeout(self.hash_timeout, manual_operation = manual_operation) 

                if manual_operation:
                    fcents_mhz = np.array([float(fcent) for fcent in self.fcents],dtype=float)
                    tbin = float(self.tbin)
                else:
                    fcents_mhz = np.array(self.meta_obs["fcents"],dtype=float)
                    tbin = self.meta_obs["tbin"]
                    
                channel_bw = 1/tbin
                fcent_hz = fcents_mhz*1e6
                
                self.log_and_post_slackmessage(f"""
                    Observation meta reports:
                    `basebands = {self.basebands}`
                    `fcents = {fcents_mhz} MHz`
                    `tbin = {tbin}`""", severity = "INFO")

                full_observation_channel_frequencies_hz = np.vstack((
                    np.arange(fcent_hz[0] - (self.nof_channels//2)*channel_bw,
                    fcent_hz[0] + (self.nof_channels//2)*channel_bw, channel_bw ),
                    np.arange(fcent_hz[1] - (self.nof_channels//2)*channel_bw,
                    fcent_hz[1] + (self.nof_channels//2)*channel_bw, channel_bw )
                ))

                full_gains_map, frequency_indices = self.correctly_place_residual_phases_and_delays(
                    ant_tune_to_collected_gains, collected_frequencies, 
                    full_observation_channel_frequencies_hz
                )

                self.log_and_post_slackmessage("""
                Plotting phase of the collected recorded gains...
                """,severity="INFO")

                phase_file_path_ac, phase_file_path_bd = plot_gain_phase(full_gains_map, full_observation_channel_frequencies_hz, fit_method = self.fit_method,
                                                                        outdir = os.path.join(self.output_dir ,"calibration_plots"), outfilestem=obs_id)

                self.log_and_post_slackmessage(f"""
                        Saved recorded gain phase for tuning AC to: 
                        `{phase_file_path_ac}`
                        and BD to:
                        `{phase_file_path_bd}`
                        """, severity = "INFO")
                
                if self.slackbot is not None:
                    try:
                        self.slackbot.upload_file(phase_file_path_ac, title =f"Recorded phases (degrees) for tuning AC from\n`{obs_id}`")
                        self.slackbot.upload_file(phase_file_path_bd, title =f"Recorded phases (degrees) for tuning BD from\n`{obs_id}`")
                    except:
                        self.log_and_post_slackmessage("Error uploading plots", severity="INFO")

                fixed_phase_filepath = redis_hget_keyvalues(self.redis_obj, CALIBRATION_CACHE_HASH)["fixed_phase"]
                try:
                    with open(fixed_phase_filepath, 'r') as f:
                        fixed_phases = json.load(f)
                except:
                    self.log_and_post_slackmessage(f"""
                        Could *not* read fixed phases from {fixed_phase_filepath} for updating with calculated residuals.
                        Clearning up and aborting calibration process...
                    """, severity = "ERROR")
                    return
                self.log_and_post_slackmessage(f"""
                Modifying fixed-phases found in
                ```{fixed_phase_filepath}```
                """)

                #calculate residual delays/phases for the collected frequencies
                if self.fit_method == "linear":
                    delay_residual_map, phase_cal_map = calc_residuals_from_polyfit(full_gains_map, full_observation_channel_frequencies_hz,frequency_indices, fixed_phases)
                elif self.fit_method == "fourier":
                    delay_residual_map, phase_cal_map = calc_residuals_from_ifft(full_gains_map,full_observation_channel_frequencies_hz, fixed_phases)

                #For json dumping:
                t_delay_dict = self.dictnpy_to_dictlist(delay_residual_map)

                #log directory for calibration delay residuals
                delay_residual_path = os.path.join(self.output_dir, "delay_residuals")
                if not os.path.exists(delay_residual_path):
                    os.makedirs(delay_residual_path)

                #-------------------------SAVE RESIDUAL DELAYS-------------------------#
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
                    """, severity = "INFO")

                #-------------------------UPDATE THE FIXED DELAYS-------------------------#
                fixed_delay_filepath = redis_hget_keyvalues(self.redis_obj, CALIBRATION_CACHE_HASH)["fixed_delay"]
                try:
                    fixed_delays = pd.read_csv(os.path.abspath(fixed_delay_filepath), names = ["IF0","IF1","IF2","IF3"],
                            header=None, skiprows=1)
                except:
                    self.log_and_post_slackmessage(f"""
                        Could *not* read fixed delays from {fixed_delay_filepath} for updating with calculated residuals.
                        Clearning up and aborting calibration process...
                    """, severity = "ERROR")
                    return
                self.log_and_post_slackmessage(f"""
                Modifying fixed-delays found in
                ```{fixed_delay_filepath}```
                with the *residual delays* calculated above in
                ```{delay_residual_filename}```
                """)
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
                if '%' in fixed_delay_filepath:
                    modified_fixed_delays_path = os.path.join((self.output_dir), ("fixed_delays/"), (os.path.splitext(os.path.basename(fixed_delay_filepath))[0].split('%')[1]+"%"+obs_id+".csv"))
                #if first time running
                else:
                    modified_fixed_delays_path = os.path.join((self.output_dir), ("fixed_delays/"), (os.path.splitext(os.path.basename(fixed_delay_filepath))[0]+"%"+obs_id+".csv"))
                
                self.log_and_post_slackmessage(f"""
                    Wrote out modified fixed delays to: 
                    ```{modified_fixed_delays_path}```
                    """, severity = "INFO")

                df = pd.DataFrame.from_dict(updated_fixed_delays)
                df.to_csv(modified_fixed_delays_path)

                #Publish new fixed delays to FEngines:
                if not self.dry_run:
                    self.log_and_post_slackmessage("""Updating fixed-delays on *all* antenna now...""", severity = "INFO")
                    self.delay_calibration.calib_csv = modified_fixed_delays_path
                    self.delay_calibration.run()
                #update the filepath for the latest fixed delay values
                if not self.dry_run:
                    redis_publish_dict_to_hash(self.redis_obj, CALIBRATION_CACHE_HASH, {"fixed_delay":modified_fixed_delays_path})
                
                #-------------------------LOAD THE NEW FIXED PHASES-------------------------#

                #bit of logic here to remove the previous filestem from the name.
                if '%' in fixed_phase_filepath:
                    modified_fixed_phases_path = os.path.join((self.output_dir), ("fixed_phases/"), (os.path.splitext(os.path.basename(fixed_phase_filepath))[0].split('%')[1]+"%"+obs_id+".json"))   
                #if first time running
                else:
                    modified_fixed_phases_path = os.path.join((self.output_dir), ("fixed_phases/"), (os.path.splitext(os.path.basename(fixed_phase_filepath))[0]+"%"+obs_id+".json"))

                self.log_and_post_slackmessage(f"""
                Wrote out modified fixed phases to: 
                ```{modified_fixed_phases_path}```""", severity = "INFO")

                if not self.dry_run:
                    self.log_and_post_slackmessage("""Updating fixed-phases on *all* antenna now...""", severity = "INFO")
                    self.update_antenna_phascals(phase_cal_map)
                    
                t_phase_cal_map = self.dictnpy_to_dictlist(phase_cal_map)
                with open(modified_fixed_phases_path, 'w+') as f:
                    json.dump(t_phase_cal_map, f)
                #update the filepath for the latest fixed phase values
                if not self.dry_run:
                    redis_publish_dict_to_hash(self.redis_obj, CALIBRATION_CACHE_HASH, {"fixed_phase":modified_fixed_phases_path})

                #-------------------------PLOT GENERATION AND SAVING-------------------------#
                delay_file_path, phase_file_path_ac, phase_file_path_bd = plot_delay_phase(delay_residual_map, phase_cal_map, 
                        full_observation_channel_frequencies_hz, outdir = os.path.join(self.output_dir ,"calibration_plots"), outfilestem=obs_id)

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
                        self.slackbot.upload_file(delay_file_path, title =f"Residual delays (ns) per antenna calculated from\n`{obs_id}`")
                        self.slackbot.upload_file(phase_file_path_ac, title =f"Phases (degrees) per frequency (Hz) for tuning AC calculated from\n`{obs_id}`")
                        self.slackbot.upload_file(phase_file_path_bd, title =f"Phases (degrees) per frequency (Hz) for tuning BD calculated from\n`{obs_id}`")
                    except:
                        self.log_and_post_slackmessage("Error uploading plots", severity="INFO")
                if manual_operation:
                    self.log_and_post_slackmessage(f"""
                    Manual calibration process run complete.
                    """)
                    return
                else:
                    #Sleep
                    self.log_and_post_slackmessage(f"""
                        Done!

                        Sleeping for {self.re_arm_time}s. 
                        Will not detect any channel triggers during this time.
                        """, severity = "INFO")
                    time.sleep(self.re_arm_time)
                    self.log_and_post_slackmessage(f"""
                        Clearing redis hash: {GPU_GAINS_REDIS_HASH} contents in anticipation of next calibration run.
                        """,severity = "DEBUG")
                    redis_clear_hash_contents(self.redis_obj, GPU_GAINS_REDIS_HASH)
            

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
    description=("""Listen for updates to GPU hashes containing calibration phases
    and generate residual delays and load calibration phases to the F-Engines.""")
    )
    parser.add_argument("--hash-timeout", type=float,default=10, required=False, help="""How long to wait for calibration 
    postprocessing to complete and update phases.""")
    parser.add_argument("--dry-run", action="store_true", help="""If run as a dry run, delay residuals and phases are 
    calcualted and written to redis/file but not loaded to the F-Engines nor applied to the existing fixed-delays.""")
    parser.add_argument("--re-arm-time", type=float, default=20, required=False, help="""After collecting phases
    from GPU nodes and performing necessary actions, the service will sleep for this duration until re-arming""")
    parser.add_argument("--fit-method", type=str, default="linear", required=False, help="""Pick the complex fitting method
    to use for residual calculation. Options are: ["linear", "fourier"]""")
    parser.add_argument("-o", "--output-dir", type=str, default="/home/cosmic/logs/calibration", required=False, help="""The output directory in 
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
    parser.add_argument('file', type=argparse.FileType('r'), nargs='*')
    args = parser.parse_args()

    input_json_dict = {}
    if len(args.file) != 0:
        for f in args.file:
            input_json_dict.update(json.load(f))
        args.no_slack_post = True
        args.dry_run = True

    slackbot = None
    if not args.no_slack_post:
        if "SLACK_BOT_TOKEN" in os.environ:
            slackbot = SlackBot(os.environ["SLACK_BOT_TOKEN"], chan_name="active_vla_calibrations", chan_id="C04KTCX4MNV")
        
        topic = f"*Logging the VLA calibration in the loop*"
        slackbot.set_topic(topic)
        slackbot.post_message(f"""
        Starting calibration observation process...""")
        
    #if input fixed delay and fixed phase files are provided, publish them to the filepath hash    
    if args.fixed_delay_to_update is not None:
        redis_publish_dict_to_hash(redis_obj, CALIBRATION_CACHE_HASH,{"fixed_delay":args.fixed_delay_to_update})
    if args.fixed_phase_to_update is not None:
        redis_publish_dict_to_hash(redis_obj, CALIBRATION_CACHE_HASH,{"fixed_phase":args.fixed_phase_to_update})

    calibrationGainCollector = CalibrationGainCollector(redis_obj, output_dir = args.output_dir, hash_timeout = args.hash_timeout, dry_run = args.dry_run,
                                re_arm_time = args.re_arm_time, fit_method = args.fit_method, slackbot = slackbot,
                                input_json_dict = None if not bool(input_json_dict) else input_json_dict, input_fcents = args.fcentmhz, input_tbin = args.tbin)
    calibrationGainCollector.start()
