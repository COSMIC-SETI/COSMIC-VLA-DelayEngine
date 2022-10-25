# COSMIC-VLA-DelayEngine
Set of scripts for the generation of geometric model delay values and calibration delay values for
delay tracking during Cosmic observations. This repository submodules the delay engine from the ATA for
use of some of its functions.

`gen_antenna_itrf.py` : Pulls local antenna coordinates from MCAST and publishes ITRF values to redis hash: META_antennaITRF. It further
                        generates a `vla_antenna_itrf.csv` file to hold these values.

`delaymodel.py`       : Inheriting from the `evla_mcast.Controller`, this process initialises off the output of `gen_antenna_itrf.py` 
                        and listens for updates of *right ascension* and *declination* values from mcast. Every period, geometric delay
                        values are calculated and published to META_modelDelays hash.
                        The model delays are calculated relative to the array center and the delays are advanced by the delay anticipated 
                        from the largest possible VLA baseline (8km).
