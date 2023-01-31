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
from cosmic.observations.slackbot import SlackBot
from cosmic.fengines import ant_remotefeng_map
from cosmic.redis_actions import redis_obj, redis_hget_keyvalues, redis_publish_dict_to_hash, redis_clear_hash_contents, redis_publish_service_pulse, redis_publish_dict_to_channel
from plot_delay_phase import plot_delay_phase

LOGFILENAME = "/home/cosmic/logs/DelayCalibration.log"
CALIBRATION_LOG_DIR = "/home/cosmic/dev/logs/calibration_logs/"
OBS_LOG_DIR = "/home/cosmic/dev/logs/obs_meta/"
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
    def __init__(self, redis_obj, fixed_delay_csv, fixed_phase_json, hash_timeout=20, re_arm_time = 30, dry_run = False, no_phase_cal = False,
    nof_streams = 4, nof_tunings = 2, nof_pols = 2, nof_channels = 1024, slackbot=None):
        self.redis_obj = redis_obj
        self.fixed_delay_csv = fixed_delay_csv
        self.fixed_phase_json = fixed_phase_json
        self.hash_timeout = hash_timeout
        self.re_arm_time = re_arm_time
        self.dry_run = dry_run
        self.no_phase_cal = no_phase_cal
        self.slackbot = slackbot
        self.nof_streams = nof_streams
        self.nof_channels = nof_channels
        self.nof_tunings = nof_tunings
        self.nof_pols = nof_pols
        self.ant_feng_map = ant_remotefeng_map.get_antennaFengineDict(self.redis_obj)
        if not self.dry_run:
            #Publish the initial fixed delays and trigger the F-Engines to load them
            self.log_and_post_slackmessage(f"""
            Publishing initial fixed-delays in
            `{self.fixed_delay_csv}`
            and fixed-phases in 
            `{self.fixed_phase_json}`
            to F-Engines.""", severity="INFO")
            self.delay_calibration = DelayCalibrationWriter(self.redis_obj, self.fixed_delay_csv)
            self.delay_calibration.run()
            with open(self.fixed_phase_json, 'r') as f:
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
        redis_publish_service_pulse(self.redis_obj, SERVICE_NAME)
        pubsub = self.redis_obj.pubsub(ignore_subscribe_messages=True)
        try:
            pubsub.subscribe(GPU_GAINS_REDIS_CHANNEL)
        except redis.RedisError:
            self.log_and_post_slackmessage(f"""
                Unable to subscribe to {GPU_GAINS_REDIS_CHANNEL} 
                channel to listen for changes to {GPU_GAINS_REDIS_HASH}.""",
            severity="ERROR")
            return False
        self.log_and_post_slackmessage("Calibration process is armed and awaiting triggers from GPU nodes.", severity="INFO")
        while True:
            #Listen for first message on subscribed channels - ignoring None
            message = pubsub.get_message(timeout=0.1)
            if message and "message" == message["type"]:
                #Get the bool data published on the first publication to that channel
                trigger = json.loads(message.get('data'))
                #fetch and calculate needed metadata:
                if trigger is not None:
                    return trigger
                else:
                    continue

    def collect_phases_for_hash_timeout(self, time_to_wait_until):
        """
        This function waits till time_to_wait expires, then collects GPU_calibrationGains,
        and processes hash contents.

        Args:
            time_to_wait_until : float, Unix time for the loop to wait until.
        Returns:
            - {<ant>_<tune_index> : [[complex(gains_pol0)], [complex(gains_pol1)]], ...}
            - collected frequency dict of {tune: [n_collected_freqs], ...}
            - list[antnames in observation]
            - obs_id of observation used for received gains
        """
        time.sleep(time_to_wait_until)

        obs_id = None

        calibration_gains = redis_hget_keyvalues(self.redis_obj, GPU_GAINS_REDIS_HASH)

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

    def calc_residual_delays_and_phases(self, ant_tune_to_collected_gains, collected_frequencies):
        """
        Taking all the concatenated phases and frequencies, use a fit of values to determine
        the residual delays and phases.

        Args: 
            ant_tune_to_collected_gains : {<ant>_<tune_index> : [[complex(gains_pol0)], [complex(gains_pol1)]], ...}
            collected frequency dict of {tune: [n_collected_freqs], ...}

        Return:
            delay_residual_map : {<ant>_<tune_index> : [[residual_delay_pol0],[residual_delay_pol1]]}, ...} in nanoseconds
            phase_residual_map : {<ant>_<tune_index> : [[residual_phase_pol0],[residual_phase_pol1]]}, ...} in radians
        """
        delay_residual_map = {}
        phase_residual_map = {} 
        amp_map = {} 

        for ant_tune, gain_matrix in ant_tune_to_collected_gains.items():
            tune = ant_tune.split('_')[1]
            tune = int(tune)
            residual_delays = np.zeros(self.nof_pols)
            residual_phases = np.zeros((self.nof_pols,len(collected_frequencies[tune])))
            
            #If there are values present in the gain matrix:
            if any(gain_matrix):
                gain_matrix = np.array(gain_matrix,dtype=np.complex64)
                t_col_frequencies = np.array(collected_frequencies[tune],dtype = float)

                phase_matrix = np.angle(gain_matrix)

                for pol in range(self.nof_pols):
                    unwrapped_phases = np.unwrap(phase_matrix[pol,:])
                    phase_slope, _ = np.polyfit(t_col_frequencies, unwrapped_phases, 1)
                    residual = unwrapped_phases - (phase_slope * t_col_frequencies)
                    residual_delays[pol] = (phase_slope / (2*np.pi)) * 1e9
                    residual_phases[pol,:] = residual % (2*np.pi)

                amp_map[ant_tune] = np.mean(np.abs(gain_matrix),axis=1)

            delay_residual_map[ant_tune] = residual_delays
            phase_residual_map[ant_tune] = residual_phases

        return delay_residual_map, phase_residual_map, amp_map

    def correctly_place_residual_phases_and_delays(self, residual_phases, residual_delays, 
        collected_frequencies, full_observation_channel_frequencies):
        """
        By investigating the placement of `collected_frequencies` inside of `full_observation_channel_frequencies_hz`,
        a map of how to place the calculated `residual_phases` inside an array of `self.nof_channels` per stream per antenna
        may be generated.

        Args:
            residual_phases: a dictionary mapping of {<ant>_<tune_index> : [phase_residual]}, ...}
            residual_delays: a dictionary mapping of {<ant>_<tune_index> : [delay_residual]}, ...}
            collected_frequences: collected frequency dict of {tune: [n_collected_freqs], ...}
            full_observation_channel_frequencies_hz: a matrix of dims(nof_tunings, nof_channels)

        Returns:
            full_residual_phase_map : a dictionary mapping of {ant: [nof_streams, nof_frequencies]} in radians
            full_residual_delay_map : a dictionary mapping of {ant: [nof_streams, 1]} in nanoseconds
        """
        full_residual_phase_map = {}
        full_residual_delay_map = {}
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
        
        log_message = """-------------------------------------------------------------"""
        try:
            log_message += f"""\n
            Calculating phase and delay residuals for tuning 0 off frequencies: 
            `{collected_frequencies[0][0]}Hz->{collected_frequencies[0][-1]}Hz`\n
            while total observation frequencies for tuning 0 are: 
            `{full_observation_channel_frequencies[0,0]}->{full_observation_channel_frequencies[0,-1]}Hz`"""
        except IndexError:
            log_message += f"""\n
            No values received for tuning 0, and so no residuals will be calculated for that tuning."""
        try:
            log_message += f"""\n
            Calculating phase and delay residuals for tuning 1 off frequencies: 
            `{collected_frequencies[1][0]}Hz->{collected_frequencies[1][-1]}Hz`\n
            while total observation frequencies for tuning 1 are: 
            `{full_observation_channel_frequencies[1,0]}Hz->{full_observation_channel_frequencies[1,-1]}Hz`"""
        except IndexError:
            log_message += f"""\n
            No values received for tuning 1, and so no residuals will be calculated for that tuning."""
        log_message += "\n-------------------------------------------------------------"
        self.log_and_post_slackmessage(log_message, severity="INFO")

        #Initialise full_residual_phase_map and full_residual_delay_map
        phase_zeros = np.zeros((self.nof_streams, self.nof_channels))
        delay_zeros = np.zeros(self.nof_streams)
        for ant in self.ants:
            full_residual_delay_map[ant] = delay_zeros.copy()
            full_residual_phase_map[ant] = phase_zeros.copy()

        #sort phases according to sorting of frequencies, and place them in nof_chan array correctly
        for ant_tune, phases in residual_phases.items():
            ant, tune = ant_tune.split('_')
            tune = int(tune)

            #per antenna, per tuning
            sorted_phases = phases[:,sortings[tune]]

            full_residual_phase_map[ant][tune*2, frequency_indices[tune]] = sorted_phases[0]
            full_residual_phase_map[ant][(tune*2)+1, frequency_indices[tune]] = sorted_phases[1]
            full_residual_delay_map[ant][tune*2] = residual_delays[ant_tune][0]
            full_residual_delay_map[ant][(tune*2)+1] = residual_delays[ant_tune][1]

        return full_residual_phase_map, full_residual_delay_map

    def start(self):
        while True:
            #Launch function that waits for first valid message:
            trigger = self.await_trigger()
            if trigger:
                self.log_and_post_slackmessage(f"""
                    Calibration process has been triggered.\n
                    Dry run = {self.dry_run}, applying phase calibrations = {not self.no_phase_cal},
                    hash timeout = {self.hash_timeout}s and re-arm time = {self.re_arm_time}s""", severity = "INFO")

                #FOR SPOOFING - TEMPORARY AND NEEDS TO BE SMARTER:
                self.basebands = [
                    "AC_8BIT",
                    "BD_8BIT"
                    ]

                #Start function that waits for hash_timeout before collecting redis hash.
                ant_tune_to_collected_gains, collected_frequencies, self.ants, obs_id = self.collect_phases_for_hash_timeout(self.hash_timeout) 

                #FOR SPOOFING - TEMPORARY AND NEEDS TO BE SMARTER:
                obs_meta_file_path = (os.path.join(OBS_LOG_DIR,obs_id+"_AC_8BIT_metadata.json") if os.path.exists(os.path.join(OBS_LOG_DIR,obs_id+"_AC_8BIT_metadata.json"))
                     else os.path.join(OBS_LOG_DIR,obs_id+"_BD_8BIT_metadata.json"))

                try:
                    with open(obs_meta_file_path) as f:
                        metadata = json.load(f)["META"]
                except:
                    self.log_and_post_slackmessage(f"""
                    Could not find any metadata logs corresponding to:
                    {obs_id}.
                    Cannot extract useful metadata required for calibration gain collation.
                    """, severity ="ERROR")
                    return

                fcent_mhz = np.array(metadata['fcents'],dtype=float)
                fcent_hz = np.array(fcent_mhz)*1e6
                tbin = float(metadata['tbin'])
                channel_bw = 1/tbin
                
                self.log_and_post_slackmessage(f"""
                    Observation meta reports:
                    `basebands = {self.basebands}`
                    `fcents = {fcent_mhz} MHz`
                    `tbin = {tbin}`""", severity = "INFO")

                #calculate residual delays/phases for the collected frequencies
                delay_residual_map, phase_residual_map, amp_map = self.calc_residual_delays_and_phases(ant_tune_to_collected_gains, collected_frequencies)

                self.log_and_post_slackmessage(f"""
                Phases collected from the GPU nodes for observation:
                `{obs_id}`
                have mean amplitudes per <ant_tuning> of:
                ```{pprint.pformat(json.dumps(self.dictnpy_to_dictlist(amp_map))).replace("'", '"')}```
                """,severity="INFO")

                full_observation_channel_frequencies_hz = np.vstack((
                    np.arange(fcent_hz[0] - (self.nof_channels//2)*channel_bw,
                    fcent_hz[0] + (self.nof_channels//2)*channel_bw, channel_bw ),
                    np.arange(fcent_hz[1] - (self.nof_channels//2)*channel_bw,
                    fcent_hz[1] + (self.nof_channels//2)*channel_bw, channel_bw )
                ))

                full_residual_phase_map, full_residual_delay_map = self.correctly_place_residual_phases_and_delays(
                    phase_residual_map, delay_residual_map, collected_frequencies, 
                    full_observation_channel_frequencies_hz
                )

                #For json dumping:
                t_delay_dict = self.dictnpy_to_dictlist(full_residual_delay_map)
                t_phase_dict = self.dictnpy_to_dictlist(full_residual_phase_map)

                #Save residual delays
                delay_filename = os.path.join(CALIBRATION_LOG_DIR,f"calibrationdelayresiduals_{obs_id}.json")
                self.log_and_post_slackmessage(f"""
                    Wrote out calculated *residual delays* to: 
                    {delay_filename}""", severity = "DEBUG")
                with open(delay_filename, 'w') as f:
                    json.dump(t_delay_dict, f)
                redis_publish_dict_to_hash(self.redis_obj, "META_residualDelays", t_delay_dict)

                pretty_print_json = pprint.pformat(json.dumps(t_delay_dict)).replace("'", '"')
                self.log_and_post_slackmessage(f"""
                    Calculated the following delay residuals from UVH5 recording
                    `{obs_id}`:

                    ```{pretty_print_json}```
                    """, severity = "INFO")

                #Save residual phases
                phase_filename = os.path.join(CALIBRATION_LOG_DIR,f"calibrationphaseresiduals_{obs_id}.json")
                self.log_and_post_slackmessage(f"""
                    Wrote out calculated *residual phases* to: 
                    {phase_filename}""", severity = "DEBUG")
                with open(phase_filename, 'w') as f:
                    json.dump(t_phase_dict, f)
                redis_publish_dict_to_hash(self.redis_obj, "META_residualPhases",t_phase_dict)
                
                #In the event of a wet run, we want to update fixed delays as well as load phase calibrations to the F-Engines
                if not self.dry_run:
                     # update phases
                    if not self.no_phase_cal:
                        try:
                            with open(self.fixed_phase_json, 'r') as f:
                                fixed_phases = json.load(f)
                        except:
                            self.log_and_post_slackmessage(f"""
                                Could *not* read fixed phases from {self.fixed_phase_json} for updating with calculated residuals.
                                Clearning up and aborting calibration process...
                            """, severity = "ERROR")
                            return
                        self.log_and_post_slackmessage(f"""
                        Modifying fixed-phases found in
                        ```{self.fixed_phase_json}```
                        with the *residual phases* calculated.
                        """)
                        updated_fixed_phases = {}
                        for ant, phases in fixed_phases.items():
                            if ant in full_residual_phase_map:
                                updated_fixed_phases[ant] = (np.array(phases,dtype=float) + full_residual_phase_map[ant]).tolist()
                            else:
                                updated_fixed_phases[ant] = phases

                        #bit of logic here to remove the previous filestem from the name.
                        if '%' in self.fixed_phase_json:
                            modified_fixed_phases_path = os.path.join(CALIBRATION_LOG_DIR+os.path.splitext(os.path.basename(self.fixed_phase_json))[0].split('%')[1]+"%"+obs_id+".json")                    
                        #if first time running
                        else:
                            modified_fixed_phases_path = os.path.join(CALIBRATION_LOG_DIR+os.path.splitext(os.path.basename(self.fixed_phase_json))[0]+"%"+obs_id+".json" )

                        self.log_and_post_slackmessage(f"""
                        Wrote out modified fixed phases to: 
                        ```{modified_fixed_phases_path}```
                        Updating fixed-phases on *all* antenna now...""", severity = "INFO")

                        self.update_antenna_phascals(updated_fixed_phases)
                        with open(modified_fixed_phases_path, 'w+') as f:
                            json.dump(updated_fixed_phases, f)

                        redis_publish_dict_to_hash(self.redis_obj, CALIBRATION_CACHE_HASH,{"fixed_phase":modified_fixed_phases_path})
                        #Overwrite the json path to the new json path for modification in the next run
                        self.fixed_phase_json = modified_fixed_phases_path

                    else:
                        self.log_and_post_slackmessage(f"""
                            Calibration process was run with argument no-phase-cal = {self.no_phase_cal}. 
                            Fixed delays *will* be updated with delay residuals but phase calibration values *will not* be written to FPGA's.
                        """, severity = "INFO")

                    # update fixed_delays
                    try:
                        fixed_delays = pd.read_csv(os.path.abspath(self.fixed_delay_csv), names = ["IF0","IF1","IF2","IF3"],
                                header=None, skiprows=1)
                    except:
                        self.log_and_post_slackmessage(f"""
                            Could *not* read fixed delays from {self.fixed_delay_csv} for updating with calculated residuals.
                            Clearning up and aborting calibration process...
                        """, severity = "ERROR")
                        return
                    self.log_and_post_slackmessage(f"""
                    Modifying fixed-delays found in
                    ```{self.fixed_delay_csv}```
                    with the *residual delays* calculated above.
                    """)
                    fixed_delays = fixed_delays.to_dict()
                    updated_fixed_delays = {}
                    for i, tune in enumerate(list(fixed_delays.keys())):
                        sub_updated_fixed_delays = {}
                        for ant, delay in fixed_delays[tune].items():
                            if ant in full_residual_delay_map:
                                sub_updated_fixed_delays[ant] = delay - full_residual_delay_map[ant][i]
                            else:
                                sub_updated_fixed_delays[ant] = delay
                        updated_fixed_delays[tune] = sub_updated_fixed_delays

                    #bit of logic here to remove the previous filestem from the name.
                    if '%' in self.fixed_delay_csv:
                        modified_fixed_delays_path = os.path.join(CALIBRATION_LOG_DIR+os.path.splitext(os.path.basename(self.fixed_delay_csv))[0].split('%')[1]+"%"+obs_id+".csv")                    
                    #if first time running
                    else:
                        modified_fixed_delays_path = os.path.join(CALIBRATION_LOG_DIR+os.path.splitext(os.path.basename(self.fixed_delay_csv))[0]+"%"+obs_id+".csv" )
                    
                    self.log_and_post_slackmessage(f"""
                        Wrote out modified fixed delays to: 
                        ```{modified_fixed_delays_path}```
                        Updating fixed-delays on *all* antenna now...""", severity = "INFO")

                    df = pd.DataFrame.from_dict(updated_fixed_delays)
                    df.to_csv(modified_fixed_delays_path)

                    #Publish new fixed delays to FEngines:
                    self.delay_calibration.calib_csv = modified_fixed_delays_path
                    self.delay_calibration.run()

                    redis_publish_dict_to_hash(self.redis_obj, CALIBRATION_CACHE_HASH,{"fixed_delay":modified_fixed_delays_path})
                    #Overwrite the csv path to the new csv path for modification in the next run
                    self.fixed_delay_csv = modified_fixed_delays_path
                
                else:
                    self.log_and_post_slackmessage("""
                        Calibration process is running in *dry-mode*. 
                        Fixed delays are *not* updated with delay residuals and 
                        phase calibration values are *not* written to FPGA's.
                    """, severity = "INFO")

                #Generate plots and save/publish them:

                delay_file_path, phase_file_path = plot_delay_phase(full_residual_delay_map,full_residual_phase_map, 
                        full_observation_channel_frequencies_hz,outdir = CALIBRATION_LOG_DIR, outfilestem=obs_id)

                self.log_and_post_slackmessage(f"""
                        Saved  residual delay plot to: 
                        `{delay_file_path}`
                        and residual phase plot to:
                        `{phase_file_path}`
                        """, severity = "DEBUG")

                if self.slackbot is not None:
                    try:
                        self.slackbot.upload_file(delay_file_path, title =f"Residual delays (ns) per antenna calculated from\n`{obs_id}`")
                        self.slackbot.upload_file(phase_file_path, title =f"Residual phases (degrees) per frequency (Hz) calculated from\n`{obs_id}`")
                    except:
                        self.log_and_post_slackmessage("Error uploading plots", severity="INFO")

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
    parser.add_argument("--no-phase-cal", action="store_true", help="""If specified, only residual delays are updated and no
    phase calibrations are applied.""")
    parser.add_argument("-f","--fixed-delay-to-update", type=str, required=False, help="""
    csv file path to latest fixed delays that must be modified by the residual delays calculated in this script. If not provided,
    process will try use fixed-delay file path in cache.""")
    parser.add_argument("-p","--fixed-phase-to-update", type=str, required=False, help="""
    json file path to latest fixed phases that must be modified by the residual phases calculated in this script. If not provided,
    process will try use fixed-phase file path in cache.""")
    parser.add_argument("--no-slack-post", action="store_true",help="""If specified, logs are not posted to slack.""")
    args = parser.parse_args()

    slackbot = None
    if not args.no_slack_post:
        if "SLACK_BOT_TOKEN" in os.environ:
            slackbot = SlackBot(os.environ["SLACK_BOT_TOKEN"], chan_name="active_vla_calibrations", chan_id="C04KTCX4MNV")
        
        topic = f"*Logging the VLA calibration in the loop*"
        slackbot.set_topic(topic)
        slackbot.post_message(f"""
        Starting calibration observation process...""")
        
    try:
        filepath_cache = redis_hget_keyvalues(redis_obj, CALIBRATION_CACHE_HASH)
    except:
        print(f"Could not fetch filepath cache from {CALIBRATION_CACHE_HASH}.")
    try:
        fixed_delay_path = filepath_cache["fixed_delay"]
    except:
        print(f"Contents of {CALIBRATION_CACHE_HASH} did not contain fixed_delay entry.")
    try:
        fixed_phase_path = filepath_cache["fixed_phase"]
    except:
        print(f"Contents of {CALIBRATION_CACHE_HASH} did not contain fixed_phase entry.")

    fixed_delay_path = args.fixed_delay_to_update if args.fixed_delay_to_update is not None else fixed_delay_path
    fixed_phase_path = args.fixed_phase_to_update if args.fixed_phase_to_update is not None else fixed_phase_path


    calibrationGainCollector = CalibrationGainCollector(redis_obj, fixed_delay_csv = fixed_delay_path, fixed_phase_json = fixed_phase_path,
                                hash_timeout = args.hash_timeout, dry_run = args.dry_run, no_phase_cal = args.no_phase_cal,
                                re_arm_time = args.re_arm_time, slackbot = slackbot)
    calibrationGainCollector.start()
