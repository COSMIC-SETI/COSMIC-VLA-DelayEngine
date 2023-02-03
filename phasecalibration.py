import json
import argparse
from cosmic.redis_actions import redis_obj, redis_publish_dict_to_hash, redis_publish_dict_to_channel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description=('Collect fixed phases from json file and publish them to "META_calibrationPhases" and trigger FEngines to load them.')
    )
    parser.add_argument("fixed_json", type=str, help="path to the latest fixed-phases json.")
    args = parser.parse_args()

    with open(args.fixed_json) as f:
        redis_publish_dict_to_hash(redis_obj, "META_calibrationPhases",json.load(f))
        redis_publish_dict_to_channel(redis_obj, "update_calibration_phases", True)