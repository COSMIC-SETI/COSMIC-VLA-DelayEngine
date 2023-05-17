import json
import os
import numpy as np
import argparse
from delaycalibration import CALIBRATION_CACHE_HASH
from cosmic.fengines import ant_remotefeng_map
from cosmic.redis_actions import redis_obj, redis_publish_dict_to_hash, redis_publish_dict_to_channel

def load_phase_calibrations(json_file):
    ant_feng_map = ant_remotefeng_map.get_antennaFengineDict(redis_obj)
    with open(json_file, 'r') as f:
        ant_phase_cal_map = json.load(f)
        redis_publish_dict_to_hash(redis_obj, "META_calibrationPhases",ant_phase_cal_map)
        redis_publish_dict_to_channel(redis_obj, "update_calibration_phases", True)
        redis_publish_dict_to_hash(redis_obj, CALIBRATION_CACHE_HASH, {"fixed_phase":json_file})
    
    ant_cal_phase_correct = {}   
    for ant, cal_phase in ant_phase_cal_map.items():
        cal_phase_correct = []
        feng = ant_feng_map[ant]
        expected_cal_phase = (np.array(cal_phase,dtype=float) + np.pi) % (2 * np.pi) - np.pi
        for stream in range(expected_cal_phase.shape[0]):
            try:
                cal_phase_correct += [bool(np.all(np.isclose(expected_cal_phase[stream,:],
                                np.array(feng.phaserotate.get_phase_cal(stream),dtype=float), atol=1e-1)))] 
            except:
                cal_phase_correct = None
        ant_cal_phase_correct[ant] = cal_phase_correct if cal_phase_correct is None else all(cal_phase_correct)
    redis_publish_dict_to_hash(redis_obj,"META_calPhasesCorrectlyLoaded",ant_cal_phase_correct)
    return ant_cal_phase_correct

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description=('Collect fixed phases from json file and publish them to "META_calibrationPhases" and trigger FEngines to load them.')
    )
    parser.add_argument("-f","--fixed-json", type=str, help="path to the latest fixed-phases json.")
    args = parser.parse_args()
    load_phase_calibrations(os.path.abspath(args.fixed_json))