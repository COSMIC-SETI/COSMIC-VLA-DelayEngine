import pandas as pd
import numpy as np
import argparse
import os
import redis
import time
import json
from cosmic.redis_actions import redis_obj, redis_hget_keyvalue, redis_hget_keyvalues, redis_publish_dict_to_hash

class CalibrationGainCollector():
    def __init__(self, redis_obj, hash_timeout=10, nof_streams = 4, nof_channels = 1024):
        self.redis_obj = redis_obj
        self.hash_timeout = hash_timeout
        self.nof_streams = nof_streams
        self.nof_channels = nof_channels
        self.pubsub = self.redis_obj.pubsub(ignore_subscribe_messages=True)
        try:
            self.pubsub.subscribe("gpu_calibrationphases")
        except redis.RedisError:
            raise redis.RedisError("""Unable to subscribe to gpu_calibrationphases channel to listen for 
            changes to GPU_calibrationPhases.""")
    
    def get_tuningidx_and_start_freq(self, message_key):
        key_split = message_key.split(',')
        tuning = key_split[-1]
        tuning_index = self.basebands.index(tuning)
        start_freq = float(key_split[0])*1e6

        return tuning_index, start_freq

    def populate_phase_cal_map(self, ant_to_phase_dict, tuning_idx):
        for ant, phase_dict in ant_to_phase_dict.items():
            freqs = phase_dict['freq_array']
            phase_0 = phase_dict['pol0_phases']
            phase_1 = phase_dict['pol1_phases']

            #Find where to place the phases in phase_cal_map
            frequency_indices = self.full_observation_channel_frequencies_hz.searchsorted(freqs)
            
            self.phase_cal_map[ant][tuning_idx*2, frequency_indices] = phase_0.copy()
            self.phase_cal_map[ant][tuning_idx*2 + 1, frequency_indices] = phase_1.copy()          

        print("Received phases for tuning: ",tuning_idx, "\n and frequency channels: ",frequency_indices)
        #assume all frequency arrays in the phase dicts are equal 
        self.collected_frequency_indices += frequency_indices.tolist()

    def collect_phases_for_hash_timeout(self, time_to_wait_until):
        while time.time() <= time_to_wait_until:
            #Listen for messages on subscribed channels
            message = self.pubsub.get_message(timeout=0.1)
            if message and "message" == message["type"]:
                #Initialise a phase calibration array for four streams of zeros:
                try:
                    hash_key = str(message.get('data'))
                except:
                    print("cannot fetch message")
                payload = redis_hget_keyvalue(self.redis_obj, "GPU_calibrationPhases", hash_key)
                tune_idx, _ = self.get_tuningidx_and_start_freq(hash_key)
                self.populate_phase_cal_map(payload, tune_idx)
        return

    def calc_residual_delays(self):
        delay_residual_map = {}
        phase_residual_map = {}

        for ant, phase_matrix in self.phase_cal_map.items():
            #Only the phases we have collected... not the ones left to zero
            phases_unwrapped = np.unwrap(phase_matrix[:,self.collected_frequency_indices],axis=1)
            residual_delays = np.zeros(self.nof_streams)
            residual_phases = np.zeros((self.nof_streams,len(self.collected_frequency_indices)))
            for stream in range(self.nof_streams):
                #Only the frequencies we've collected
                freqs_collected = self.full_observation_channel_frequencies_hz[self.collected_frequency_indices]
                phase_slope, _ = np.polyfit(freqs_collected, phases_unwrapped[stream,:], 1)
                residual = phases_unwrapped[stream,:] - (phase_slope * freqs_collected)
                residual_delays[stream]=phase_slope / (2*np.pi)
                residual_phases[stream,:] = residual % (2*np.pi)
            delay_residual_map[ant] = residual_delays.tolist()
            phase_residual_map[ant] = residual_phases.tolist()
        return delay_residual_map, phase_residual_map

    def start(self):
        while True:
            #Listen for messages on subscribed channels
            message = self.pubsub.get_message(timeout=0.1)
            if message and "message" == message["type"]:
                #Initialise a phase calibration array for four streams of zeros:
                try:
                    hash_key = str(message.get('data'))
                except:
                    print("cannot fetch message")
                payload = redis_hget_keyvalue(self.redis_obj, "GPU_calibrationPhases", hash_key)
                metadata = redis_hget_keyvalues(self.redis_obj, "META")
                self.basebands = metadata.get('baseband')
                tune_idx, _ = self.get_tuningidx_and_start_freq(hash_key)
                self.collected_frequency_indices = []

                self.ants = list(payload.keys())
                #create the phase_cal map of antnames -> phase values of shape (nof_streams, nof_channels)
                self.phase_cal_map ={}
                zeros = np.zeros((self.nof_streams,self.nof_channels))
                for ant in self.ants:
                    self.phase_cal_map[ant] = zeros.copy()
                fcent_mhz = metadata['fcents'][tune_idx]
                print("Observation center frequency: ",fcent_mhz,"MHz")
                fcent_hz = fcent_mhz * 1e6
                #spoofing fcents
                fcent_hz = np.median(payload[self.ants[0]]['freq_array'])
                print("Observation center frequency: ",fcent_hz,"Hz")
                channel_width_hz = payload[self.ants[0]]['freq_array'][1] - payload[self.ants[0]]['freq_array'][0]
                print("Channel width: ",channel_width_hz,"Hz")
                start_observation_channel_frequencies_hz = fcent_hz - (self.nof_channels//2)*channel_width_hz
                stop_observation_channel_frequencies_hz = fcent_hz + (self.nof_channels//2)*channel_width_hz

                self.full_observation_channel_frequencies_hz = np.arange(start_observation_channel_frequencies_hz, stop_observation_channel_frequencies_hz, channel_width_hz)
                print("Full observation channel frequncies: ",self.full_observation_channel_frequencies_hz,"\n of size: ",self.full_observation_channel_frequencies_hz.size)
                
                self.populate_phase_cal_map(payload, tune_idx)

                #Listen for new messages for the hash timeout period
                time_to_wait_until = time.time() + self.hash_timeout
                self.collect_phases_for_hash_timeout(time_to_wait_until) 

                #calculate residual delays/phases, save them and publish them
                delay_residual_map, phase_residual_map = self.calc_residual_delays()
                filename = os.path.join("/home/cosmic/dev/logs/calibration_logs/",f"calibrationdelayresiduals_{round(float(metadata['tend']))}_{metadata['src']}.json")
                with open(filename, 'w') as f:
                    json.dump(delay_residual_map, f)
                redis_publish_dict_to_hash(self.redis_obj, "META_residualDelays",delay_residual_map)

                filename = os.path.join("/home/cosmic/dev/logs/calibration_logs/",f"calibrationphaseresiduals_{round(float(metadata['tend']))}_{metadata['src']}.json")
                with open(filename, 'w') as f:
                    json.dump(phase_residual_map, f)
                redis_publish_dict_to_hash(self.redis_obj, "META_residualPhases",phase_residual_map)

                #load phases to F-Engines
                

                # update fixed_delays
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description=("""Listen for updates to GPU hashes containing calibration phases
    and generate residual delays and load calibration phases to the F-Engines.""")
    )
    parser.add_argument("--hash-timeout", type=float,default=10, required=False, help="""How long to wait for calibration 
    postprocessing to complete and update phases.""")
    args = parser.parse_args()

    calibrationGainCollector = CalibrationGainCollector(redis_obj, hash_timeout = args.hash_timeout)
    calibrationGainCollector.start()