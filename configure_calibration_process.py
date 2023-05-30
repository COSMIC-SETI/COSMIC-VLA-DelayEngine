from cosmic.redis_actions import redis_obj, redis_publish_dict_to_hash
from delaycalibration import load_delay_calibrations
from phasecalibration import load_phase_calibrations
from calibration_gain_collator import CONFIG_HASH
from cosmic.observations.slackbot import SlackBot
import argparse
import json
import os

def load_and_configure_calibrations(hash_timeout=20, re_arm_time = 30, fit_method = "linear",
                                    input_fixed_delays = "fixed_delay_init.csv", input_fixed_phases = "fixed_phases_init.json",
                                    snr_threshold = 4.0, slackbot = None):
    input_fixed_delays = os.path.abspath(input_fixed_delays)
    input_fixed_phases = os.path.abspath(input_fixed_phases)
    if os.path.isfile(input_fixed_delays) and os.path.isfile(input_fixed_phases):
        config_dict={
            "hash_timeout":hash_timeout,
            "re_arm_time":re_arm_time,
            "fit_method":fit_method,
            "input_fixed_delays":input_fixed_delays,
            "input_fixed_phases":input_fixed_phases,
            "snr_threshold":snr_threshold
        }
        redis_publish_dict_to_hash(redis_obj, CONFIG_HASH, config_dict)
        load_delay_calibrations(input_fixed_delays)
        load_phase_calibrations(input_fixed_phases)
        msg = f"""
        Reconfiguring calibration process with following configuration:
        ```{config_dict}```"""
        print(msg)
        if slackbot is not None:
            slackbot.post_message(msg)
    else:
        print("Provided fixed phases and delays files do not exist. Aborting.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        """Configure the calibration process by populating the redis hash
        CAL_configuration for use in the next calibration process loop."""
    )
    parser.add_argument("--hash-timeout", type=float,default=10, required=False, help="""How long to wait for calibration 
    postprocessing to complete and update phases.""")
    parser.add_argument("--re-arm-time", type=float, default=20, required=False, help="""After collecting phases
    from GPU nodes and performing necessary actions, the service will sleep for this duration until re-arming""")
    parser.add_argument("--fit-method", type=str, default="fourier", required=False, help="""Pick the complex fitting method
    to use for residual calculation. Options are: ["linear", "fourier"]""")
    parser.add_argument("-f","--fixed-delay-to-update", type=str, required=False, help="""
    csv file path to latest fixed delays that must be modified by the residual delays calculated in this script. If not provided,
    process will try use fixed-delay file path in cache.""")
    parser.add_argument("-p","--fixed-phase-to-update", type=str, required=False, help="""
    json file path to latest fixed phases that must be modified by the residual phases calculated in this script. If not provided,
    process will try use fixed-phase file path in cache.""")
    parser.add_argument("--no-slack-post", action="store_true",help="""If specified, logs are not posted to slack.""")
    parser.add_argument("--snr-threshold", type=float, default = 4.0, required=False, 
                        help="""The snr threshold above which the process will reject applying the calculated delay
                        and phase residual calibration values""")
    args = parser.parse_args()

    slackbot = None
    if not args.no_slack_post:
        if "SLACK_BOT_TOKEN" in os.environ:
            slackbot = SlackBot(os.environ["SLACK_BOT_TOKEN"], chan_name="active_vla_calibrations", chan_id="C04KTCX4MNV")

    load_and_configure_calibrations(hash_timeout=args.hash_timeout, re_arm_time=args.re_arm_time,
                                    fit_method=args.fit_method, input_fixed_delays=args.fixed_delay_to_update,
                                    input_fixed_phases=args.fixed_phase_to_update, snr_threshold=args.snr_threshold, slackbot=slackbot)
    