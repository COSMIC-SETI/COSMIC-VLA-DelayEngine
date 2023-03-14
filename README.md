# COSMIC-VLA-DelayEngine
This repository contains all the scripts used for delay calculation, monitoring and control. All communication with the F-Engines and GPU-node services is done via [remoteobjects](https://github.com/MydonSolutions/remoteobjects-py) and Redis.

The entirety of the delay tracking and calibration process is viewed as follows (only Cosmic Headnode processes are contained within this repository):

![Delay Management System](https://user-images.githubusercontent.com/28049678/224703031-b299c177-adc0-4045-8214-1863d2c255f4.jpg)

Pale blue blocks are Head node processes, light rust-red blocks are FPGA node processes and light pink blocks are GPU node processes. While this Readme will touch on the delay-specific processes from the GPU and FPGA nodes used in Delay management, a full explanation of their operation is found at their respective repositories: [vla-dev](https://github.com/realtimeradio/vla-dev) and [COSMIC-VLA-CalibrationEngine](https://github.com/COSMIC-SETI/COSMIC-VLA-CalibrationEngine).

The above image may be separated into two main processes. Delay coefficient and delay/phase calibration calculation.

### The Delay Model:
`delaymodel.py` which runs as a service on the headnode (delaymode.service) is responsible for calculating the Geometric delay coefficients needed for delay tracking. It listens on two redis channels: `meta_antennaproperties` and `obs_phase_center` for updates on antenna positions and updates on source pointings respectively.

The module makes use of the submoduled repository from the ATA's geometric delay module [delayengine](https://github.com/wfarah/delay_engine), to calculate delay coefficients.

![Geometric delay tracking](https://user-images.githubusercontent.com/28049678/224711832-590c9a4c-c651-4b8f-ab5b-05e72c4eeb2a.jpg)

Along with the reception of source pointing updates on `obs_phase_center` is an optional `loadtime` in micro-seconds. This interface is available to enable exact source-phasing in time. Loadtime can be left as `None`. In the light purple block above, the decision structure for the `delaymodel` is shown. If a new pointing is received or it has been longer than 5s since the last set of delay coefficients were sent out, the received/last recieved `loadtime` is evaluated and delay ceofficients are generated for the appropriate phase centre while `fpga_loadtime` is updated.

Interpolation within the delay model is done over 3s and delay coefficients are generated to 3rd order. These coefficients are in nanoseconds where the geometric delay at time `t_i` may be calculated as:

`Delay(t_i) = delay_ns + (delay_rate_nsps * t_i) + (0.5 * delay_raterate_nsps2 * ti^2)`

Additional values for phase correction are provided by the `delaymodel`, namely `effective_lo` and `sideband` which are VLA tuning specific values provided for the phase compensation of the frequency downsampling upstream in the VLA digitiser.

The delay coefficients for each antenna are sent out by the `delaymodel` as a dictionary on redis channels with name `<antname>_delays` where <antname> is the name of the antenna for which those geometric delays apply.

The light red block shows the interpolation and loading process performed on the FPGA nodes by an antenna-specific `CosmicFengine` instance. Each instance listens on `<antname>_delays`, `update_calibration_delays` and `update_calibration_phases` channels. `<antname>_delays` is how the delay tracking thread receives its geometric delay coefficients, loadtime instructions and phase correction factors, whereas the two `update*` channels are channels used by the delay calibration process to alert the delay tracking thread of updates to phase and delay calibration value updates.

All model delay values are then also loaded to redis hash `META_modelDelays` for logging and monitoring purposes.

If delay coefficients are received with a `loadtime` 1-2s into the future, delay coefficients are updated to those received, delay values are calculated for the `loadtime` provided and set to load to the F-Engine at that `loadtime`. If not, the old delay coefficients are retained and delay values are calculated for `0.5s` into the future and set to load then.


After loading, values are checked and sent to the redis hash `FENG_delayStatus` for logging and monitoring purposes.

### The Delay Calibration Model:
`calibration_gain_collator.py` which runs as a service on the headnode (calibration_gain_collator.service) is responsible for collecting the gain values for each frequency that span all GPU nodes. This design is depicted below:

![DelayCalibrationProcess](https://user-images.githubusercontent.com/28049678/225007506-85b9f608-2d98-4e48-9a1d-6a15422e90f5.jpg)

The image is divided into two halves. Blue half and white half. The white half contains processes from outside this repository running on the GPU and Head nodes - namely [calibrate_uvh5.py](https://github.com/COSMIC-SETI/COSMIC-VLA-CalibrationEngine/blob/main/calibrate_uvh5.py) and [postprocess_hub.py](https://github.com/COSMIC-SETI/COSMIC-VLA-PythonLibs/blob/main/scripts/postprocess_hub.py).

The blue half is entirely the `calibration_gain_collator.py` process.

When a calibration recording is about to begin, a message with field "postprocess" set to "calibrate-uvh5" is sent out on Redis channel "observations" is sent to indicate so. Contained in the message are also the `project_id` and `dataset_id` for the observation. This message which serves as a trigger, moves the `calibration_gain_collator` to retrieve from the Redis hash `META`, the `fcent`, `tbin` and `source` fields. 

After the calibration recording is complete, the postprocessor will launch `calibrate_uvh5.py` and direct it to extract gains from the recently recorded *.uvh5 files. This happens in parallel across all GPU instances. Each *.uvh5 file contains a subset of frequencies and an individual tuning (2 polarisations). As gains cannot be calculated for all frequency channels/streams off of a subset, all gains must be collated and ordered.

This introduces the need for the Headnode service `calibration_gain_collator.py`. All `calibrate_uvh5.py` processes publish their gains results to `GPU_calibrationGains` under the unique key: <startingfrequency>,<tuning_id>. With each publication, the `calibrate_uvh5.py` process will send out a boolean trigger on `gpu_calibrationgains` to indicate a publication has been made. On the first trigger from this channel, the `calibration_gain_collator.py` will start a timer (user defined). Once that timer expires, all gains will be collected from `GPU_calibrationGains`.

Then using the information collected from the "META" hash and "observations" channel message, the `calibration_gain_collator.py` will start placing gains in their correct place amongst 1024 channels centered on fcents.

After this sorting, the gains per antenna, per stream, per frequency are sent to one of the calibration fitting kernals inside `calibration_residual_kernals.py`. What is returned from these kernels is a set of residual delays and calibration phases. The residual delays are subtracted from the previous calibration delays and then loaded to the F-Engines along with the calibration phases. 

The `calibration_gain_collator.py` process updates "CAL_fixedValuePaths" hash to reflect that those files contain the present fixed values on the F-Engines and then goes back to an armed state. 
