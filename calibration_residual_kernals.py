import numpy as np

def calc_residuals_from_polyfit(ant_to_gains, observation_frequencies, frequency_indices):
        """
        Accept mapping of antenna to gains along with the observation frequencies of dimension (n_tunings, n_chans). In the event 
        not all gains are received, a start and stop demarcate the region over which to calculate the fit.
        Returns ant to residual delay and ant to residual phase maps.

        Args: 
            ant_to_gains : {<ant> : [[complex(gains_pol0_tune0)], [complex(gains_pol1_tune0)], [complex(gains_pol0_tune1)], [complex(gains_pol1_tune1)]], ...}
            observation_frequencies : list of dimension (n_tunings, nchans) in Hz
            frequency_indices : indices of the collected gains in the full n_chans per tuning

        Return:
            delay_residual_map : {<ant> : [[residual_delay_pol0_tune0], [residual_delay_pol1_tune0], [residual_delay_pol0_tune1], [residual_delay_pol1_tune1]]}, ...} in nanoseconds
            phase_residual_map : {<ant> : [[residual_phase_pol0_tune0],[residual_phase_pol1_tune0],
                                        [residual_phase_pol0_tune1],[residual_phase_pol1_tune1]]}, ...} in radians
        """
        delay_residual_map = {}
        phase_residual_map = {} 

        for ant, gain_matrix in ant_to_gains.items():
            gain_matrix = np.array(gain_matrix, dtype = np.complex64)
            nof_streams = int(gain_matrix.shape[0])
            nof_tunings = int(observation_frequencies.shape[0])
            nof_pols = int(nof_streams/nof_tunings)
            residual_delays = np.zeros(nof_streams,dtype=np.float64)
            residual_phases = np.zeros(gain_matrix.shape,dtype=np.float64)

            for tune in range(nof_tunings):
                if frequency_indices[tune].size==0:
                    #This means that nothing was collected for that tuning
                    continue
                else:
                    freq_range = observation_frequencies[tune, frequency_indices[tune]]
                    phases = np.angle(gain_matrix[:,frequency_indices[tune]])
                    for pol in range(nof_pols):  #probably unecessary - could do with some matrix mult stuff
                        #some binary logic
                        stream_idx = int(str(tune)+str(pol),2)
                        unwrapped_phases = np.unwrap(phases[stream_idx,:])
                        phase_slope = np.polyfit(freq_range, unwrapped_phases,1)[0]
                        residuals = unwrapped_phases - (phase_slope*freq_range)
                        residual_delays[stream_idx] = (phase_slope / (2*np.pi)) * 1e9
                        residual_phases[stream_idx,frequency_indices[tune]] = residuals % (2*np.pi)
            
            delay_residual_map[ant] = residual_delays
            phase_residual_map[ant] = residual_phases

        return delay_residual_map, phase_residual_map

def calc_residuals_from_ifft(ant_to_gains, observation_frequencies):
    """
    Accept mapping of antenna to gains.
    Returns ant to residual delay and ant to residual phase maps.

    Args:
        ant_to_gains : {<ant> : [[complex(gains_pol0_tune0)], [complex(gains_pol1_tune0)], [complex(gains_pol0_tune1)], [complex(gains_pol1_tune1)]], ...}
        observation_frequencies : list of dimension (n_tunings, nchans) in Hz
    Return:
        delay_residual_map : {<ant> : [[residual_delay_pol0_tune0], [residual_delay_pol1_tune0], [residual_delay_pol0_tune1], [residual_delay_pol1_tune1]]}, ...} in nanoseconds
        phase_residual_map : {<ant> : [[residual_phase_pol0_tune0],[residual_phase_pol1_tune0],
                                        [residual_phase_pol0_tune1],[residual_phase_pol1_tune1]]}, ...} in radians
    """
    delay_residual_map = {}
    phase_residual_map = {}
    for ant, gain_matrix in ant_to_gains.items():
        gain_matrix = np.array(gain_matrix, dtype = np.complex64)
        nof_streams = gain_matrix.shape[0]
        nof_tunings,nof_chan = observation_frequencies.shape
        nof_pols = int(nof_streams/nof_tunings)
        ifft_abs_matrix = np.abs(np.fft.ifft(gain_matrix, axis = 1))
        max_idxs = np.argmax(ifft_abs_matrix,axis=1)
        residual_delays = np.zeros(nof_streams,dtype=np.float64)
        residual_phases = np.zeros(gain_matrix.shape,dtype=np.float64)

        for tune in range(nof_tunings):
            chan_width = observation_frequencies[tune,1] - observation_frequencies[tune,0]
            freqs = np.fft.fftfreq(nof_chan, chan_width)
            tlags = freqs*1e9
            for pol in range(nof_pols):   #probably unecessary - could do with some matrix mult stuff
                #some binary logic
                stream_idx = int(str(tune)+str(pol),2)
                residual_delays[stream_idx] = -1.0 * tlags[max_idxs[stream_idx]]
                gain_from_residual_delay = np.exp(2j*np.pi*(observation_frequencies[tune,:]*1e-9)*residual_delays[stream_idx])
                zero_gains_indices = np.where(gain_matrix[stream_idx,:]==0.0)
                residual_phases[stream_idx] = np.angle(gain_matrix[stream_idx,:]/gain_from_residual_delay)
                residual_phases[stream_idx, zero_gains_indices] = 0.0

        delay_residual_map[ant] = residual_delays
        phase_residual_map[ant] = residual_phases

    return delay_residual_map, phase_residual_map
