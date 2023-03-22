import json
import os
import argparse
from delaycalibration import CALIBRATION_CACHE_HASH
from cosmic.redis_actions import redis_obj, redis_publish_dict_to_hash, redis_publish_dict_to_channel

def load_phase_calibrations(json_file):
    with open(json_file, 'r') as f:
        redis_publish_dict_to_hash(redis_obj, "META_calibrationPhases",json.load(f))
        redis_publish_dict_to_channel(redis_obj, "update_calibration_phases", True)
        redis_publish_dict_to_hash(redis_obj, CALIBRATION_CACHE_HASH, {"fixed_phase":json_file})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description=('Collect fixed phases from json file and publish them to "META_calibrationPhases" and trigger FEngines to load them.')
    )
    parser.add_argument("-f","--fixed-json", type=str, help="path to the latest fixed-phases json.")
    args = parser.parse_args()
    load_phase_calibrations(os.path.abspath(args.fixed_json))