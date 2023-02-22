import numpy as np
import matplotlib.pyplot as plt
import argparse
import threading
import os
import time
from cosmic.redis_actions import redis_obj, redis_hget_keyvalues

class delayWatcher:
    def __init__(self, args, redis_obj):
        self.time_granularity = 0.5
        self.redis_obj = redis_obj
        self.antenna = args.antenna
        self.duration = args.duration
        file_path = os.path.dirname(os.path.realpath(__file__))
        seps = os.sep+os.altsep if os.altsep else os.sep
        self.output_directory = os.path.join(file_path,
                        os.path.splitdrive(args.output_directory)[1].lstrip(seps))
        meta = redis_hget_keyvalues(self.redis_obj, "META")
        self.src = meta.get('src')
        self.ra_deg = meta.get('ra_deg')
        self.dec_deg = meta.get('dec_deg')

        self.watch_model_event = threading.Event()
        self.model_delay={}
        for ant in self.antenna:
            self.model_delay[ant] = [[],[],[],[]]
        
        self.model_time = []
        
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

    def watch_model(self):
        while self.watch_model_event.is_set():
            t = time.time()
            delay_coeffs = redis_hget_keyvalues(self.redis_obj, "META_modelDelays", self.antenna)
            for ant in self.antenna:
                calibration_delay = np.fromiter(redis_hget_keyvalues(self.redis_obj, "META_calibrationDelays",ant)[ant].values(),dtype=float)
                for i in range(4):
                    self.model_delay[ant][i] += [calibration_delay[i] + delay_coeffs[ant]["delay_ns"]]
            self.model_time += [delay_coeffs[self.antenna[0]]["time_value"]]
            tdiff = time.time()-t
            time.sleep(self.time_granularity-tdiff)

    def monitor_fengs(self):
        ant_to_delayvector = {}
        ant_to_phasevector = {}
        ant_to_loadtimevector = {}
        for ant in self.antenna:
            ant_to_delayvector[ant] = [[],[]]
            ant_to_phasevector[ant] = [[],[]]
            ant_to_loadtimevector[ant] = []

        start = time.time()
        future = start + self.duration
        while time.time() <= future:

            meta = redis_hget_keyvalues(self.redis_obj, "META")
            if self.src != meta.get('src') or self.ra_deg != meta.get('ra_deg') or self.dec_deg != meta.get('dec_deg'): 
                print("Source has changed during recording. Stopping delay watching prematurely")
                break

            t = time.time()
            delaystates = redis_hget_keyvalues(self.redis_obj, "FENG_delayStatus", self.antenna)
            for ant, delaystate in delaystates.items():
                ant_to_delayvector[ant][0]+=[delaystate["expected_delay_ns"]]
                ant_to_delayvector[ant][1]+=[delaystate["firmware_delay_ns"]]
                ant_to_phasevector[ant][0]+=[delaystate["expected_phase_rad"]]
                ant_to_phasevector[ant][1]+=[delaystate["firmware_phase_rad"]]
                ant_to_loadtimevector[ant] += [delaystate["delays_loaded_at"]]
            tdiff = time.time()-t
            time.sleep(self.time_granularity-tdiff)
        return ant_to_delayvector, ant_to_phasevector, ant_to_loadtimevector,start

    def record(self):
        self.watch_model_thread = threading.Thread(
            target=self.watch_model, args=(), daemon=False
        )
        self.watch_model_event.set()
        self.watch_model_thread.start()
        ant_to_delayvector, ant_to_phasevector, ant_to_loadtimevector, start = self.monitor_fengs()
        self.watch_model_event.clear()
        self.watch_model_thread.join()

        modeltimevector = np.array(self.model_time) - start
        for ant in self.antenna:
            fig, axs = plt.subplots(4, 2, sharex = True, sharey=False, constrained_layout=True,
                    figsize = (18,12))
            delay_vector = np.array(ant_to_delayvector[ant])
            phase_vector = np.array(ant_to_phasevector[ant])
            loadtimevector = np.array(ant_to_loadtimevector[ant]) - start
            for i in range(4):
                average_deltadelay = np.mean(np.diff(delay_vector[0,:,i]))
                average_deltaphase = np.mean(np.diff(phase_vector[0,:,i]))
                # axs[i,0].plot(loadtimevector, delay_vector[0,:,i], '.' label='expected')
                axs[i,0].plot(loadtimevector, delay_vector[1,:,i], label='firmware')
                axs[i,0].plot(modeltimevector, self.model_delay[ant][i], '.', label='model')
                axs[i,0].legend(loc = 'upper right')
                axs[i,0].set_title(f"Delay tracking stream {i}\nmean delta = {average_deltadelay}ns")
                axs[i,0].set_ylabel("Delay (ns)")
                # axs[i,1].plot(loadtimevector, phase_vector[0,:,i], '.' label='expected')
                axs[i,1].plot(loadtimevector, phase_vector[1,:,i], '.', label='firmware')
                axs[i,1].legend(loc = 'upper right')
                axs[i,1].set_title(f"Phase tracking stream {i}\nmean delta = {average_deltaphase}rad/pi")
                axs[i,1].set_ylabel("Phase (rad/pi)")

            fig.suptitle(f"""Delay and phase tracking over {args.duration}s
                        for antenna {ant} and source {self.src}""")
            fig.supxlabel("Time (s)")

            plt.savefig(os.path.join(self.output_directory,f"{ant}_delayphase_tracking.png"), dpi = 300)
            plt.close()    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = """Provided with a time and antenna set, produce plots
        of the phase and delay over time for those antenna."""
    )
    parser.add_argument('-a', '--antenna', nargs="*", default = ['ea02'],
                        required = True, help="Specify list of antenna to watch")
    parser.add_argument('-t', '--duration', type=float, default = 10,
                        required = True, help="Specify the duration to watch phase and delay for" )
    parser.add_argument('-o', '--output-directory', type=str, default = '.', required = False,
                        help = "The output directory to which to save the generated plots")
    args = parser.parse_args()

    delayWatcher = delayWatcher(args, redis_obj)
    delayWatcher.record()

