import numpy as np
import time
import argparse
import matplotlib.pyplot as plt
from cosmic.redis_actions import redis_obj, redis_hget_keyvalues

def interpolate(args):
    future = time.time() + args.duration
    ant_to_interpdelayvector = []
    ant_to_delayvector = []
    ant_to_timevector = []
    x_ax_interpvect = []
    delay_coeffs = redis_hget_keyvalues(redis_obj, "META_modelDelays")
    init_time = delay_coeffs[args.antenna]["time_value"]
    ra=delay_coeffs['deg_ra']
    dec=delay_coeffs['deg_dec']
    while time.time() <= future:
        delay_coeffs = redis_hget_keyvalues(redis_obj, "META_modelDelays")
        if ra != delay_coeffs['deg_ra'] or dec != delay_coeffs['deg_dec']:
            print("Source has changed during recording. Stopping delay watching prematurely")
            break 
        t = time.time()
        # ant_to_delayvector += [delay_coeffs[args.antenna]["delay_ns"] + delay_coeffs[args.antenna]["delay_rate_nsps"] * t]
        ant_to_delayvector += [delay_coeffs[args.antenna]["delay_ns"]]
        ant_to_timevector += [delay_coeffs[args.antenna]["time_value"]- init_time] 
        interp_till = t + args.interp_length
        while t <= interp_till:
            t_to_interp_to = time.time() - delay_coeffs[args.antenna]["time_value"]
            x_ax_interpvect += [time.time() - init_time]
            delay_to_load = (np.array([0.5 * delay_coeffs[args.antenna]["delay_raterate_nsps2"] * (t_to_interp_to**2) +
                            delay_coeffs[args.antenna]["delay_rate_nsps"] * t_to_interp_to +
                            delay_coeffs[args.antenna]["delay_ns"]],dtype=float))
            delay_rate_to_load = np.array([delay_coeffs[args.antenna]["delay_raterate_nsps2"] * t_to_interp_to +
                                        delay_coeffs[args.antenna]["delay_rate_nsps"]],dtype=float)

            tdiff = time.time()-t
            time.sleep(args.granularity - tdiff)
            t = time.time()
            ant_to_interpdelayvector += [delay_to_load]

    plt.plot(ant_to_timevector, ant_to_delayvector, '.', x_ax_interpvect, ant_to_interpdelayvector)
    plt.title(f"Delay tracking interpolation behaviour over {args.duration} for antenna {args.antenna}")
    plt.xlabel("time (s)")
    plt.ylabel("Delay (ns)")
    plt.savefig(f"{args.antenna}_delay_tracking.png", dpi = 150)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = """Provided with a time and antenna set, produce plots
        of the phase and delay over time for those antenna."""
    )
    parser.add_argument('-a', '--antenna', default = ['ea02'],
                        required = True, help="Specify antenna to watch")
    parser.add_argument('-t', '--duration', type=float, default = 10,
                        required = True, help="Specify the duration to watch phase and delay for" )
    parser.add_argument('-i', '--interp_length', type = float, default = 5, 
                        required = True, help="Specify the interpolation length in seconds")
    parser.add_argument('-n', '--granularity', type = float, default = 0.5,
                        required = False, help="""Specify the time granularity at which delay coeffs are 
                        recalculated""")
    args = parser.parse_args()

    interpolate(args)