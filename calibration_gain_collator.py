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
from cosmic.redis_actions import redis_obj, redis_hget_keyvalues, redis_publish_dict_to_hash, redis_clear_hash_contents, redis_publish_service_pulse
from plot_delay_phase import plot_delay_phase

LOGFILENAME = "/home/cosmic/logs/DelayCalibration.log"
CALIBRATION_LOG_DIR = "/home/cosmic/dev/logs/calibration_logs/"
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
        if not self.dry_run:
            self.init_antenna_phascals(0.0, ants = None)
            #Publish the initial fixed delays and trigger the F-Engines to load them
            self.log_and_post_slackmessage(f"""
            Publishing initial fixed-delays in:
            `{self.fixed_csv}`
            to F-Engines.""", severity="INFO")
            self.delay_calibration = DelayCalibrationWriter(self.redis_obj, self.fixed_csv)
            self.delay_calibration.run()
    
    def get_tuningidx_and_start_freq(self, message_key):
        key_split = message_key.split(',')
        tuning = key_split[-1]
        tuning_index = self.basebands.index(tuning)
        start_freq = float(key_split[0])*1e6

        return tuning_index, start_freq

    def init_antenna_phascals(self, init_val, ants = None):
        if ants is not None:
            ants = ants
        else:
            ants = list(self.ant_feng_map.keys())
        
        self.log_and_post_slackmessage(f"""
        Initialising phasecalibration values to {init_val} on antenna:
        {ants}\n""", severity="DEBUG")
        for ant in ants:
            feng = self.ant_feng_map[ant]
            for stream in range(self.nof_streams):
                feng.phaserotate.set_phase_cal(
                                            stream,
                                            [init_val]*1024
                                        ) 
    
    @staticmethod
    def dictnpy_to_dictlist(dictnpy):
        dictlst = {}
        for key in dictnpy:
            dictlst[key] = dictnpy[key].tolist()
        return dictlst

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
            - filestem of uvh5 file used for received gains
        """
        time.sleep(time_to_wait_until)

        filestem = None

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
            filestem_t = os.path.splitext(payload['filestem'])[0]
            if filestem is not None and filestem_t != filestem:
                self.log_and_post_slackmessage(f"""
                    Skipping {start_freq_tune} payload since it contains differing filestem {filestem_t}
                    to previously encountered filestem {filestem}.""", severity="WARNING")
                continue
            else:
                filestem = filestem_t
            
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
        
        return ant_tune_to_collected_gain, collected_frequencies, ants, filestem

    def calc_residual_delays_and_phases(self, ant_tune_to_collected_gains, collected_frequencies):
        """
        Taking all the concatenated phases and frequencies, use a fit of values to determine
        the residual delays and phases.

        Args: 
            ant_tune_to_collected_gains : {<ant>_<tune_index> : [[complex(gains_pol0)], [complex(gains_pol1)]], ...}
            collected frequency dict of {tune: [n_collected_freqs], ...}

        Return:
            delay_residual_map : {<ant>_<tune_index> : [[residual_delay_pol0],[residual_delay_pol1]]}, ...}
            phase_residual_map : {<ant>_<tune_index> : [[residual_phase_pol0],[residual_phase_pol1]]}, ...}
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
                    residual_delays[pol] = phase_slope / (2*np.pi)
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
        
        print(len(collected_frequencies[0]))
        print(len(collected_frequencies[1]))
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
            redis_publish_service_pulse(self.redis_obj, SERVICE_NAME)
            #Launch function that waits for first valid message:
            trigger = self.await_trigger()
            if trigger:
                self.log_and_post_slackmessage(f"""
                    Calibration process has been triggered.\n
                    Dry run = {self.dry_run}, applying phase calibrations = {not self.no_phase_cal},
                    hash timeout = {self.hash_timeout}s and re-arm time = {self.re_arm_time}s""", severity = "INFO")

                #Fetch and calculate needed metadata
                self.log_and_post_slackmessage("Collecting present VLA mcast observation metadata", severity = "INFO")
                try:
                    metadata = redis_hget_keyvalues(self.redis_obj, "META")
                except:
                    self.log_and_post_slackmessage("Could not collect present VLA mcast metadata. Ignoring trigger...", severity = "ERROR")
                    continue
                # #FOR SPOOFING:
                self.basebands = [
                    "AC_8BIT",
                    "BD_8BIT"
                    ]
                fcent_mhz = [
                3000.0,
                3005.0
                ]
                tbin = 1e-6

                # self.basebands = metadata.get('baseband')
                # fcent_mhz = np.array(metadata['fcents'],dtype=float)
                fcent_hz = np.array(fcent_mhz)*1e6
                # tbin = float(metadata['tbin'])
                channel_bw = 1/tbin
                
                self.log_and_post_slackmessage(f"""
                    Observation meta reports:
                    `basebands = {self.basebands}`
                    `fcents = {fcent_mhz} MHz`
                    `tbin = {tbin}`""", severity = "INFO")

                #Start function that waits for hash_timeout before collecting redis hash.
                ant_tune_to_collected_gains, collected_frequencies, self.ants, filestem = self.collect_phases_for_hash_timeout(self.hash_timeout) 

                #calculate residual delays/phases for the collected frequencies
                delay_residual_map, phase_residual_map, amp_map = self.calc_residual_delays_and_phases(ant_tune_to_collected_gains, collected_frequencies)

                self.log_and_post_slackmessage(f"""
                Phases collected from the GPU nodes for uvh5 stem:
                `{filestem}`
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
                delay_filename = os.path.join(CALIBRATION_LOG_DIR,f"calibrationdelayresiduals_{filestem}.json")
                self.log_and_post_slackmessage(f"""
                    Wrote out calculated *residual delays* to: 
                    {delay_filename}""", severity = "DEBUG")
                with open(delay_filename, 'w') as f:
                    json.dump(t_delay_dict, f)
                redis_publish_dict_to_hash(self.redis_obj, "META_residualDelays", t_delay_dict)

                pretty_print_json = pprint.pformat(json.dumps(t_delay_dict)).replace("'", '"')
                self.log_and_post_slackmessage(f"""
                    Calculated the following delay residuals from UVH5 recording
                    `{filestem}`:

                    ```{pretty_print_json}```
                    """, severity = "INFO")

                #Save residual phases
                phase_filename = os.path.join(CALIBRATION_LOG_DIR,f"calibrationphaseresiduals_{filestem}.json")
                self.log_and_post_slackmessage(f"""
                    Wrote out calculated *residual phases* to: 
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
                            #Check if ant is one of the ones we computed phases for:
                            if ant in full_residual_phase_map:
                                for stream in range(self.nof_streams):
                                    try:
                                        feng.phaserotate.set_phase_cal(
                                            stream,   
                                            full_residual_phase_map[ant][stream,:].tolist()
                                        )
                                    except:
                                        self.log_and_post_slackmessage(f"""
                                        Could *not* write out phase calibrations to the antenna: {ant}"""
                                        ,severity="ERROR")
                            #Zero the if not:
                            else:
                                try:
                                    self.init_antenna_phascals(0.0,ants = [ant])
                                except:
                                    self.log_and_post_slackmessage(f"""
                                    Could *not* zero-out phase calibrations of antenna: {ant}"""
                                    ,severity="WARNING")
                    
                    else:
                        self.log_and_post_slackmessage(f"""
                            Calibration process was run with argument no-phase-cal = {self.no_phase_cal}. 
                            Fixed delays *will* be updated with delay residuals but phase calibration values *will not* be written to FPGA's.
                        """, severity = "INFO")

                    # update fixed_delays
                    try:
                        fixed_delays = pd.read_csv(os.path.abspath(self.fixed_csv), names = ["IF0","IF1","IF2","IF3"],
                                header=None, skiprows=1)
                    except:
                        self.log_and_post_slackmessage(f"""
                            Could *not* read fixed delays from {self.fixed_csv} for updating with calculated residuals.
                            Clearning up and aborting calibration process...
                        """, severity = "ERROR")
                    self.log_and_post_slackmessage(f"""
                    Modifying fixed-delays found in
                    ```{self.fixed_csv}```
                    with the *residual delays* calculated above.
                    """)
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
                        modified_fixed_delays_path = os.path.splitext(os.path.basename(self.fixed_csv))[0].split('%')[1]+"%"+filestem+".csv"                    
                    #if first time running
                    else:
                        modified_fixed_delays_path = os.path.join(CALIBRATION_LOG_DIR+os.path.splitext(os.path.basename(self.fixed_csv))[0]+"%"+filestem+".csv" )
                    
                    self.log_and_post_slackmessage(f"""
                        Wrote out modified fixed delays to: 
                        ```{modified_fixed_delays_path}```
                        Updating fixed-delays on *all* antenna now...""", severity = "INFO")

                    df = pd.DataFrame.from_dict(updated_fixed_delays)
                    df.to_csv(modified_fixed_delays_path)

                    #Publish new fixed delays to FEngines:
                    self.delay_calibration.calib_csv = modified_fixed_delays_path
                    self.delay_calibration.run()

                    #Overwrite the csv path to the new csv path for modification in the next run
                    self.fixed_csv = modified_fixed_delays_path
                
                else:
                    self.log_and_post_slackmessage("""
                        Calibration process is running in *dry-mode*. 
                        Fixed delays are *not* updated with delay residuals and 
                        phase calibration values are *not* written to FPGA's.
                    """, severity = "INFO")

                #Generate plots and save/publish them:

                delay_file_path, phase_file_path = plot_delay_phase(full_residual_delay_map,full_residual_phase_map, 
                full_observation_channel_frequencies_hz,outdir = CALIBRATION_LOG_DIR, outfilestem=filestem)

                self.log_and_post_slackmessage(f"""
                        Saved  residual delay plot to: 
                        `{delay_file_path}`
                        and residual phase plot to:
                        `{phase_file_path}`
                        """, severity = "DEBUG")

                slackbot.upload_file(delay_file_path, title =f"Residual delays per antenna calculated from\n`{filestem}`")
                slackbot.upload_file(phase_file_path, title =f"Residual phases per frequency calculated from\n`{filestem}`")

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