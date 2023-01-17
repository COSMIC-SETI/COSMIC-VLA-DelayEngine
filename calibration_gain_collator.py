import pandas as pd
import numpy as np
import itertools
import argparse
import os
import redis
import time
import json
from delaycalibration import DelayCalibrationWriter
from cosmic.fengines import ant_remotefeng_map
from cosmic.redis_actions import redis_obj, redis_hget_keyvalues, redis_publish_dict_to_hash, redis_clear_hash_contents

GPU_PHASES_REDIS_HASH = "GPU_calibrationPhases"

class CalibrationGainCollector():
    def __init__(self, redis_obj, fixed_csv, hash_timeout=20, re_arm_time = 30, dry_run = False, nof_streams = 4, nof_tunings = 2, nof_pols = 2, nof_channels = 1024):
        self.redis_obj = redis_obj
        self.fixed_csv = fixed_csv
        self.hash_timeout = hash_timeout
        self.re_arm_time = re_arm_time
        self.dry_run = dry_run
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

    def await_trigger(self):
        pubsub = self.redis_obj.pubsub(ignore_subscribe_messages=True)
        try:
            pubsub.subscribe("gpu_calibrationphases")
        except redis.RedisError:
            raise redis.RedisError("""Unable to subscribe to gpu_calibrationphases channel to listen for 
        changes to GPU_calibrationPhases.""")
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
            print(f"Processing tuning {tune_idx}, start freq {start_freq}...")
            filestem_t = payload['filestem']
            if filestem is not None and filestem_t != filestem:
                print(f"""Skipping {start_freq_tune} payload since it contains differing filestem {filestem_t}
                to previously encountered filestem {filestem}.""")
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

        print("-------------------------------------------------------------")
        print(f"Calculated phase and delay residuals for tuning 0 off frequencies: {collected_frequencies[0][0]}:{collected_frequencies[0][-1]}Hz")
        print(f"Calculated phase and delay residuals for tuning 1 off frequencies: {collected_frequencies[1][0]}:{collected_frequencies[1][-1]}Hz")
        print("-------------------------------------------------------------")

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
                #Fetch and calculate needed metadata
                metadata = redis_hget_keyvalues(self.redis_obj, "META")
                # self.basebands = metadata.get('baseband')
                self.basebands = [
                    "AC_8BIT",
                    "BD_8BIT"
                    ]
                print("Observation meta reports basebands: ",self.basebands)
                # fcent_mhz = np.array(metadata['fcents'],dtype=float)
                fcent_mhz = [
                    2477.0,
                    3501.0
                    ]
                print("Observation meta reports fcents: ",fcent_mhz,"MHz")
                fcent_hz = np.array(fcent_mhz)*1e6
                # tbin = float(metadata['tbin'])
                tbin = 1e-6
                channel_bw = 1/tbin
                print(f"Expected channel bandwidth: {channel_bw}Hz")

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
                print("Wrote out calculated residual delays to: ",delay_filename)
                with open(delay_filename, 'w') as f:
                    json.dump(t_delay_dict, f)
                redis_publish_dict_to_hash(self.redis_obj, "META_residualDelays", t_delay_dict)

                #Save residual phases
                phase_filename = os.path.join("/home/cosmic/dev/logs/calibration_logs/",f"calibrationphaseresiduals_{filestem}.json")
                print("Wrote out calculated residual phases to: ",phase_filename)
                with open(phase_filename, 'w') as f:
                    json.dump(t_phase_dict, f)
                redis_publish_dict_to_hash(self.redis_obj, "META_residualPhases",t_phase_dict)
                
                #In the event of a wet run, we want to update fixed delays as well as load phase calibrations to the F-Engines
                if not self.dry_run:
                    #load phases to F-Engines
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
                                    print("Could not write out phase calibrations to the antenna: ",ant)
                            #Zero the rest:
                            else:
                                try:
                                    feng.phaserotate.set_phase_cal(
                                        stream,
                                        [0.0]*1024
                                    )
                                except:
                                    print("Could not write out zeros to the antenna: ",ant)

                    # update fixed_delays
                    fixed_delays = pd.read_csv(os.path.abspath(self.fixed_csv), names = ["IF0","IF1","IF2","IF3"],
                                header=None, skiprows=1)
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
                    
                    print("Wrote out modified fixed delays to: ",modified_fixed_delays_path)
                    df = pd.DataFrame.from_dict(updated_fixed_delays)
                    df.to_csv(modified_fixed_delays_path)

                    #Publish the new fixed delays and trigger the F-Engines to load them
                    delay_calibration = DelayCalibrationWriter(self.redis_obj, modified_fixed_delays_path)
                    delay_calibration.run()

                    #Overwrite the csv path to the new csv path for modification in the next run
                    self.fixed_csv = modified_fixed_delays_path

                #Sleep
                time.sleep(self.re_arm_time)
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
    parser.add_argument("-f","--fixed-delay-to-update", type=str, required=True, help="""
    csv file path to latest fixed delays that must be modified by the residual delays calculated in this script.""")
    args = parser.parse_args()

    calibrationGainCollector = CalibrationGainCollector(redis_obj, fixed_csv = args.fixed_delay_to_update, 
                                hash_timeout = args.hash_timeout, dry_run = args.dry_run, re_arm_time = args.re_arm_time)
    calibrationGainCollector.start()