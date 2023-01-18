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
from cosmic.redis_actions import redis_obj, redis_hget_keyvalues, redis_publish_dict_to_hash, redis_clear_hash_contents

LOGFILENAME = "/home/cosmic/logs/DelayCalibration.log"
logger = logging.getLogger('calibration_delays')
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

GPU_PHASES_REDIS_HASH = "GPU_calibrationPhases"

class CalibrationGainCollector():
    def __init__(self, redis_obj, fixed_csv, hash_timeout=20, re_arm_time = 30, dry_run = False, no_phase_cal = False,
    nof_streams = 4, nof_tunings = 2, nof_pols = 2, nof_channels = 1024, slackbot=None):
        self.redis_obj = redis_obj
        self.fixed_csv = fixed_csv
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
    
    def get_tuningidx_and_start_freq(self, message_key):
        key_split = message_key.split(',')
        tuning = key_split[-1]
        tuning_index = self.basebands.index(tuning)
        start_freq = float(key_split[0])*1e6

        return tuning_index, start_freq

    def log_and_post_slackmessage(self, message, severity = "INFO"):
        if severity =="INFO":
            logger.info(message)
            self.slackbot.post_message(dedent("\
                INFO:\n" + message))
        if severity =="WARN":
            logger.warn(message)
            self.slackbot.post_message(dedent("\
                WARNING:\n" + message))
        if severity =="ERROR":
            logger.error(message)
            self.slackbot.post_message(dedent("\
                ERROR:\n" + message))
        if severity == "DEBUG":
            logger.debug(message)

    def await_trigger(self):
        pubsub = self.redis_obj.pubsub(ignore_subscribe_messages=True)
        try:
            pubsub.subscribe("gpu_calibrationphases")
        except redis.RedisError:
            self.log_and_post_slackmessage("""
                Unable to subscribe to gpu_calibrationphases 
                channel to listen for changes to GPU_calibrationPhases.""",
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
        This function waits till time_to_wait expires, then collects GPU_calibrationPhases,
        and processes hash contents.

        Args:
            time_to_wait_until : float, Unix time for the loop to wait until.
        Returns:
            - {<ant>_<tune_index> : [phases], ...}
            - collected frequency dict of {tune: [n_collected_freqs], ...}
            - list[antnames in observation]
        """
        time.sleep(time_to_wait_until)

        filestem = None

        calibration_phases = redis_hget_keyvalues(self.redis_obj, GPU_PHASES_REDIS_HASH)

        #Which antenna have been provided - assume unchanging given timeout expired
        ants = list(calibration_phases[list(calibration_phases.keys())[0]].keys())
        ants.remove('filestem')
        #Initialise some empty dicts and arrays for population during this process
        collected_frequencies = {0:[],1:[]}
        ant_tune_to_collected_phase = {}
        for ant,tuning_idx in itertools.product(ants, range(self.nof_tunings)):
            ant_tune_to_collected_phase[ant+f"_{tuning_idx}"] = [[],[]]

        for start_freq_tune, payload in calibration_phases.items():
            tune_idx, start_freq = self.get_tuningidx_and_start_freq(start_freq_tune)
            self.log_and_post_slackmessage(f"Processing tuning {tune_idx}, start freq {start_freq}...", severity="DEBUG")
            filestem_t = payload['filestem']
            if filestem is not None and filestem_t != filestem:
                self.log_and_post_slackmessage(f"""
                    Skipping {start_freq_tune} payload since it contains differing filestem {filestem_t}
                    to previously encountered filestem {filestem}.""", severity="WARNING")
                continue
            else:
                del payload['filestem']
                filestem = filestem_t
            
            for ant, phase_dict in payload.items():
                key = ant+"_"+str(tune_idx)
                ant_tune_to_collected_phase[key][0] += phase_dict['pol0_phases'] 
                ant_tune_to_collected_phase[key][1] += phase_dict['pol1_phases']
                
                if not any(f in collected_frequencies[tune_idx] for f in phase_dict['freq_array']):
                    collected_frequencies[tune_idx] += phase_dict['freq_array']    
        
        return ant_tune_to_collected_phase, collected_frequencies, ants, filestem

    def calc_residual_delays_and_phases(self, ant_tune_to_collected_phase, collected_frequencies):
        """
        Taking all the concatenated phases and frequencies, use a fit of values to determine
        the residual delays and phases.

        Args: 
            ant_tune_to_collected_phase : {<ant>_<tune_index> : [phases], ...}
            collected frequency dict of {tune: [n_collected_freqs], ...}

        Return:
            delay_residual_map : {<ant>_<tune_index> : [[residual_delay_pol0],[residual_delay_pol1]]}, ...}
            phase_residual_map : {<ant>_<tune_index> : [[residual_phase_pol0],[residual_phase_pol1]]}, ...}
        """
        delay_residual_map = {}
        phase_residual_map = {} 

        for ant_tune, phase_matrix in ant_tune_to_collected_phase.items():
            tune = ant_tune.split('_')[1]
            tune = int(tune)
            residual_delays = np.zeros(self.nof_pols)
            residual_phases = np.zeros((self.nof_pols,len(collected_frequencies[tune])))
            
            #If there are values present in the phase matrix:
            if any(phase_matrix):
                phase_matrix = np.array(phase_matrix)
                t_col_frequencies = np.array(collected_frequencies[tune],dtype = float)

                for pol in range(self.nof_pols):
                    unwrapped_phases = np.unwrap(phase_matrix[pol,:])
                    phase_slope, _ = np.polyfit(t_col_frequencies, unwrapped_phases, 1)
                    residual = unwrapped_phases - (phase_slope * t_col_frequencies)
                    residual_delays[pol] = phase_slope / (2*np.pi)
                    residual_phases[pol,:] = residual % (2*np.pi)

            delay_residual_map[ant_tune] = residual_delays
            phase_residual_map[ant_tune] = residual_phases

        return delay_residual_map, phase_residual_map

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
            full_residual_phase_map : a dictionary mapping of {ant: [nof_streams, nof_frequencies]} 
            full_residual_delay_map : a dictionary mapping of {ant: [nof_streams, 1]} 
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
            {collected_frequencies[0][0]}Hz->{collected_frequencies[0][-1]}Hz\n
            while total observation frequencies for tuning 0 are: 
            {full_observation_channel_frequencies[0,0]}->{full_observation_channel_frequencies[0,-1]}Hz"""
        except IndexError:
            log_message += f"""\n
            No values received for tuning 0, and so no residuals will be calculated for that tuning."""
        try:
            log_message += f"""\n
            Calculating phase and delay residuals for tuning 1 off frequencies: 
            {collected_frequencies[1][0]}Hz->{collected_frequencies[1][-1]}Hz\n
            while total observation frequencies for tuning 0 are: 
            {full_observation_channel_frequencies[1,0]}Hz->{full_observation_channel_frequencies[1,-1]}Hz"""
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
                    Dry run = {self.dry_run}, applying phase calibrations = {self.no_phase_cal},
                    hash timeout = {self.hash_timeout}s and re-arm time = {self.re_arm_time}s""", severity = "INFO")

                #Fetch and calculate needed metadata
                self.log_and_post_slackmessage("Collecting present VLA mcast observation metadata", severity = "INFO")
                try:
                    metadata = redis_hget_keyvalues(self.redis_obj, "META")
                except:
                    self.log_and_post_slackmessage("Could not collect present VLA mcast metadata. Ignoring trigger...", severity = "ERROR")
                    continue
                #FOR SPOOFING:
                # self.basebands = [
                #     "AC_8BIT",
                #     "BD_8BIT"
                #     ]
                # fcent_mhz = [
                #     2477.0,
                #     3501.0
                #     ]
                # tbin = 1e-6

                self.basebands = metadata.get('baseband')
                fcent_mhz = np.array(metadata['fcents'],dtype=float)
                fcent_hz = np.array(fcent_mhz)*1e6
                tbin = float(metadata['tbin'])
                channel_bw = 1/tbin
                
                self.log_and_post_slackmessage(f"""
                    Observation meta reports:
                    basebands = {self.basebands}
                    fcents = {fcent_mhz} MHz
                    tbin = {tbin}""", severity = "INFO")

                #Start function that waits for hash_timeout before collecting redis hash.
                ant_tune_to_collected_phase, collected_frequencies, self.ants, filestem = self.collect_phases_for_hash_timeout(self.hash_timeout) 

                #calculate residual delays/phases for the collected frequencies
                delay_residual_map, phase_residual_map = self.calc_residual_delays_and_phases(ant_tune_to_collected_phase, collected_frequencies)

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
                t_delay_dict = {}
                t_phase_dict = {}
                for ant, val in full_residual_phase_map.items():
                    t_delay_dict[ant] = full_residual_delay_map[ant].tolist() 
                    t_phase_dict[ant] = val.tolist() 

                #Save residual delays
                delay_filename = os.path.join("/home/cosmic/dev/logs/calibration_logs/",f"calibrationdelayresiduals_{filestem}.json")
                self.log_and_post_slackmessage(f"""
                    Wrote out calculated residual delays to: 
                    {delay_filename}""", severity = "DEBUG")
                with open(delay_filename, 'w') as f:
                    json.dump(t_delay_dict, f)
                redis_publish_dict_to_hash(self.redis_obj, "META_residualDelays", t_delay_dict)

                pretty_print_json = pprint.pformat(json.dumps(t_delay_dict)).replace("'", '"')
                self.log_and_post_slackmessage(f"""
                    Calculated the following delay residuals from UVH5 recording
                    {filestem}:

                    ```{pretty_print_json}```
                    """, severity = "INFO")

                #Save residual phases
                phase_filename = os.path.join("/home/cosmic/dev/logs/calibration_logs/",f"calibrationphaseresiduals_{filestem}.json")
                self.log_and_post_slackmessage(f"""
                    Wrote out calculated residual phases to: 
                    {phase_filename}""", severity = "DEBUG")
                with open(phase_filename, 'w') as f:
                    json.dump(t_phase_dict, f)
                redis_publish_dict_to_hash(self.redis_obj, "META_residualPhases",t_phase_dict)
                
                #In the event of a wet run, we want to update fixed delays as well as load phase calibrations to the F-Engines
                if not self.dry_run:
                    #load phases to F-Engines
                    if not self.no_phase_cal:
                        self.log_and_post_slackmessage(f"""
                        Loading phase calibration results to Antenna `{self.ants}`
                        and zeroing phase calibration values in all other antenna...
                        """,severity="INFO")
                        for ant, feng in self.ant_feng_map.items():
                            for stream in range(self.nof_streams):
                                #Check if ant is one of the ones we computed phases for:
                                if ant in full_residual_phase_map:
                                    try:
                                        feng.phaserotate.set_phase_cal(
                                            stream,   
                                            full_residual_phase_map[ant][stream,:].tolist()
                                        )
                                    except:
                                        self.log_and_post_slackmessage(f"""
                                        Could not write out phase calibrations to the antenna: {ant}"""
                                        ,severity="ERROR")
                                #Zero the rest:
                                else:
                                    try:
                                        feng.phaserotate.set_phase_cal(
                                            stream,
                                            [0.0]*1024
                                        )
                                    except:
                                        self.log_and_post_slackmessage(f"""
                                        Could not zero-out phase calibrations of antenna: {ant}"""
                                        ,severity="WARNING")
                    
                    else:
                        self.log_and_post_slackmessage(f"""
                            Calibration process was run with argument no-phase-cal = {self.no_phase_cal}. 
                            Fixed delays will be updated with delay residuals but phase calibration values will not be written to FPGA's.
                        """, severity = "INFO")

                    # update fixed_delays
                    try:
                        fixed_delays = pd.read_csv(os.path.abspath(self.fixed_csv), names = ["IF0","IF1","IF2","IF3"],
                                header=None, skiprows=1)
                    except:
                        self.log_and_post_slackmessage(f"""
                            Could not read fixed delays from {self.fixed_csv} for updating with calculated residuals.
                            Clearning up and aborting calibration process.
                        """, severity = "ERROR")
                    fixed_delays = fixed_delays.to_dict()
                    updated_fixed_delays = {}
                    for i, tune in enumerate(list(fixed_delays.keys())):
                        sub_updated_fixed_delays = {}
                        for ant, delay in fixed_delays[tune].items():
                            if ant in full_residual_delay_map:
                                sub_updated_fixed_delays[ant] = delay + float(-1e9 * full_residual_delay_map[ant][i])
                            else:
                                sub_updated_fixed_delays[ant] = delay
                        updated_fixed_delays[tune] = sub_updated_fixed_delays

                    #bit of logic here to remove the previous filestem from the name.
                    if '%' in self.fixed_csv:
                        modified_fixed_delays_path = "/home/cosmic/dev/logs/calibration_logs/"+os.path.basename(self.fixed_csv).split('.')[0].split('%')[1]+"%"+filestem+".csv"                    
                    #if first time running
                    else:
                        modified_fixed_delays_path = "/home/cosmic/dev/logs/calibration_logs/"+os.path.basename(self.fixed_csv).split('.')[0]+"%"+filestem+".csv" 
                    
                    self.log_and_post_slackmessage(f"""
                        Wrote out modified fixed delays to: 
                        ```{modified_fixed_delays_path}```
                        Updating fixed-delays on all antenna now...""", severity = "INFO")

                    df = pd.DataFrame.from_dict(updated_fixed_delays)
                    df.to_csv(modified_fixed_delays_path)

                    #Publish the new fixed delays and trigger the F-Engines to load them
                    delay_calibration = DelayCalibrationWriter(self.redis_obj, modified_fixed_delays_path)
                    delay_calibration.run()

                    #Overwrite the csv path to the new csv path for modification in the next run
                    self.fixed_csv = modified_fixed_delays_path
                
                else:
                    self.log_and_post_slackmessage("""
                        Calibration process is running in dry-mode. 
                        Fixed delays are not updated with delay residuals and 
                        phase calibration values are not written to FPGA's.
                    """, severity = "INFO")

                #Sleep
                self.log_and_post_slackmessage(f"""
                    Done!

                    Sleeping for {self.re_arm_time}s. 
                    Will not detect any channel triggers during this time.
                    """, severity = "INFO")
                time.sleep(self.re_arm_time)
                self.log_and_post_slackmessage(f"""
                    Clearing redis hash: {GPU_PHASES_REDIS_HASH} contents in anticipation of next calibration run.
                    """,severity = "DEBUG")
                redis_clear_hash_contents(self.redis_obj, GPU_PHASES_REDIS_HASH)
            

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
    parser.add_argument("-f","--fixed-delay-to-update", type=str, required=True, help="""
    csv file path to latest fixed delays that must be modified by the residual delays calculated in this script.""")
    args = parser.parse_args()

    slackbot = None
    if "SLACK_BOT_TOKEN" in os.environ:
        slackbot = SlackBot(os.environ["SLACK_BOT_TOKEN"], chan_name="active_vla_calibrations", chan_id="C04KTCX4MNV")
    
    topic = f"*Logging the VLA calibration in the loop*"
    slackbot.set_topic(topic)
    slackbot.post_message(f"""
    Starting calibration observation process with starting fixed-delays:
    {args.fixed_delay_to_update}""")


    calibrationGainCollector = CalibrationGainCollector(redis_obj, fixed_csv = args.fixed_delay_to_update, 
                                hash_timeout = args.hash_timeout, dry_run = args.dry_run, no_phase_cal = args.no_phase_cal,
                                re_arm_time = args.re_arm_time, slackbot = slackbot)
    calibrationGainCollector.start()