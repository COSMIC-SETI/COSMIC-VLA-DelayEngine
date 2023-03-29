import numpy as np
from scipy.stats import median_absolute_deviation as mad

def calc_residuals_from_polyfit(ant_to_gains, observation_frequencies, current_phase_cals, frequency_indices, delay_residual_rejection=100):
        """
        Accept mapping of antenna to gains along with the observation frequencies of dimension (n_tunings, n_chans). In the event 
        not all gains are received, a start and stop demarcate the region over which to calculate the fit.
        Returns ant to residual delay and ant to phase cal maps.

        Args: 
            ant_to_gains : A dictionary mapping of antenna name to complex gain matrix of dimension (n_streams, n_chans).
                    {<ant> : [[complex(gains_pol0_tune0)], [complex(gains_pol1_tune0)], [complex(gains_pol0_tune1)], [complex(gains_pol1_tune1)]], ...}
            observation_frequencies : list of frequencies in Hz of dimension (n_tunings, nchans)
            current_phase_cals : A mapping of antenna to phase coefficients (in radians) on the F-Engine at present - a matrix of dimension (n_streams, n_chans).
                    {<ant> : [[phase_cal_pol0_tune0], [phase_cal_pol1_tune0], [phase_cal_pol0_tune1], [phase_cal_pol1_tune1]], ...}
            frequency_indices : indices of the collected gains (sorted) in the full n_chans per tuning: 
                            {tuning_idx : np.array(int)}
            delay_residual_rejection float: The absolute delay residual threshold in nanoseconds above which the process will reject applying the calculated delay
                                and phase residual calibration values

        Return:
            delay_residual_map : A mapping of antenna to delay residual values in nanoseconds of dimension (n_streams)
                    {<ant> : [residual_delay_pol0_tune0, residual_delay_pol1_tune0, residual_delay_pol0_tune1, residual_delay_pol1_tune1]}, ...}
            phase_cal_map : A mapping of antenna to phase calibration value in radians of dimension (n_streams, n_chans).
                    {<ant> : [[phase_pol0_tune0],[phase_pol1_tune0],
                                [phase_pol0_tune1],[phase_pol1_tune1]]}, ...}
    """
        delay_residual_map = {}
        phase_cal_map = {} 

        for ant, phase_matrix in current_phase_cals.items():
            phase_matrix = np.array(phase_matrix,dtype=float)
            if ant in ant_to_gains.keys():

                gain_matrix = np.array(ant_to_gains[ant], dtype = np.complex64)
                #Subtract the last applied phases from the gain (incase incorrect)
                current_phase_matrix = np.exp(1j * np.array(phase_matrix))
                new_gain_matrix = gain_matrix * current_phase_matrix

                nof_streams = int(gain_matrix.shape[0])
                nof_tunings = int(observation_frequencies.shape[0])
                nof_pols = int(nof_streams/nof_tunings)
                residual_delays = np.zeros(nof_streams,dtype=np.float64)
                phase_cals = np.zeros(gain_matrix.shape,dtype=np.float64)

                for tune in range(nof_tunings):
                    if frequency_indices[tune].size==0:
                        #This means that nothing was collected for that tuning
                        continue
                    else:
                        freq_range = observation_frequencies[tune, frequency_indices[tune]]
                        phases = np.angle(new_gain_matrix[:,frequency_indices[tune]])
                        for pol in range(nof_pols):  #probably unecessary - could do with some matrix mult stuff
                            #some binary logic
                            stream_idx = int(str(tune)+str(pol),2)
                            unwrapped_phases = np.unwrap(phases[stream_idx,:])
                            phase_slope = np.polyfit(freq_range, unwrapped_phases,1)[0]
                            residuals = unwrapped_phases - (phase_slope*freq_range)
                            residual_delay = (phase_slope / (2*np.pi)) * 1e9
                            if np.abs(residual_delay) > delay_residual_rejection:
                                residual_delays[stream_idx] = 0.0
                                phase_cals[stream_idx] = phase_matrix[stream_idx]
                            residual_delays[stream_idx] = (phase_slope / (2*np.pi)) * 1e9
                            phase_cals[stream_idx,frequency_indices[tune]] = residuals % (2*np.pi)
                
                delay_residual_map[ant] = residual_delays
                phase_cal_map[ant] = phase_cals
            else:
                phase_cal_map[ant] = phase_matrix
                delay_residual_map[ant] = np.zeros(nof_streams,dtype=np.float64)

        return delay_residual_map, phase_cal_map

def calc_residuals_from_ifft(ant_to_gains, observation_frequencies, current_phase_cals, frequency_indices, delay_residual_rejection=100):
    """
    Accept mapping of antenna to gains.
    Returns ant to residual delay and ant to phase cal maps.

    Args:
        ant_to_gains : A dictionary mapping of antenna name to complex gain matrix of dimension (n_streams, n_chans).
                    {<ant> : [[complex(gains_pol0_tune0)], [complex(gains_pol1_tune0)], [complex(gains_pol0_tune1)], [complex(gains_pol1_tune1)]], ...}
        observation_frequencies : list of frequencies in Hz of dimension (n_tunings, nchans)
        current_phase_cals : A mapping of antenna to phase coefficients on the F-Engine at present - a matrix of dimension (n_streams, n_chans).
                    {<ant> : [[phase_cal_pol0_tune0], [phase_cal_pol1_tune0], [phase_cal_pol0_tune1], [phase_cal_pol1_tune1]], ...}
        frequency_indices : indices of the collected gains (sorted) in the full n_chans per tuning: 
                            {tuning_idx : np.array(int)}
        delay_residual_rejection float: The absolute delay residual threshold in nanoseconds above which the process will reject applying the calculated delay
                                and phase residual calibration values

    Return:
        delay_residual_map : A mapping of antenna to delay residual values in nanoseconds of dimension (n_streams)
                    {<ant> : [residual_delay_pol0_tune0, residual_delay_pol1_tune0, residual_delay_pol0_tune1, residual_delay_pol1_tune1]}, ...}
        phase_cal_map : A mapping of antenna to phase calibration value in radians of dimension (n_streams, n_chans).
                    {<ant> : [[phase_pol0_tune0],[phase_pol1_tune0],
                                        [phase_pol0_tune1],[phase_pol1_tune1]]}, ...} in radians
    """
    delay_residual_map = {}
    phase_cal_map = {}
    
    #Finding the number of antennas here. Maybe there is a better way
    #nant = len(ant_to_gains.keys())
    #ant_ind, Also get the antenna index somehow here starting from zero

    for ant, phase_matrix in current_phase_cals.items():
        phase_matrix = np.array(phase_matrix,dtype=float)
        if ant in ant_to_gains.keys():
            gain_matrix = np.array(ant_to_gains[ant], dtype = np.complex64)

            #Subtract the last applied phases from the gain (incase incorrect)
            current_phase_matrix = np.exp(1j * np.array(phase_matrix))
            new_gain_matrix = gain_matrix * current_phase_matrix

            nof_streams = gain_matrix.shape[0]
            nof_tunings,nof_chan = observation_frequencies.shape
            nof_pols = int(nof_streams/nof_tunings)
            ifft_abs_matrix = np.abs(np.fft.ifft(new_gain_matrix, axis = 1))
            max_idxs = np.argmax(ifft_abs_matrix,axis=1)
            
            """
            Adding some lines to do the SNR of the peak here
            sigma = mad(ifft_abs_matrix,axis=1)
            median = np.median(ifft_abs_matrix,axis=1)
            """
            residual_delays = np.zeros(nof_streams,dtype=np.float64)
            phase_cals = np.zeros(gain_matrix.shape,dtype=np.float64)
            #SNR = np.zeros((nant, nof_streams),dtype=np.float64) #array to store SNR
            #sigma_phase = np.zeros((nant, nof_streams),dtype=np.float64) #Array to store spread of phases
            for tune in range(nof_tunings):
                #find range outside collected gains
                uncollected_gain_range = np.setdiff1d(np.arange(observation_frequencies[tune,:].size), frequency_indices[tune], assume_unique = True)
                chan_width = observation_frequencies[tune,1] - observation_frequencies[tune,0]
                freqs = np.fft.fftfreq(nof_chan, chan_width)
                tlags = freqs*1e9
                for pol in range(nof_pols):   #probably unecessary - could do with some matrix mult stuff
                    #some binary logic
                    stream_idx = int(str(tune)+str(pol),2)
                    
                    #Calculating the power here and SNR
                    #signal_pow = ifft_abs_matrix[stream_idx, max_idxs[stream_idx]]
                    #SNR[antenna_index, stream_idex] = (signal_pow - median[stream_idx])/sigma[stream_idx]

                    residual_delay = -1.0 * tlags[max_idxs[stream_idx]]
                    #Now rather than using the abs(residual_delay) value, use
                    #if SNR[antenna_index, stream_idex] > 4.0, then update the delay values, if not there is no delay peak and probably np.argmax will 
                    #pick up some random noise and no point in updating the delay values
                    
                    if np.abs(residual_delay) > delay_residual_rejection:
                        residual_delays[stream_idx] = 0.0
                        phase_cals[stream_idx] = phase_matrix[stream_idx]
                    else:
                        residual_delays[stream_idx] = residual_delay
                        gain_from_residual_delay = np.exp(2j*np.pi*(observation_frequencies[tune,:]*1e-9)*residual_delays[stream_idx])
                        phase_cals[stream_idx] = np.angle(new_gain_matrix[stream_idx,:]/gain_from_residual_delay)
                        #zero all phases outside the collected gains range
                        phase_cals[stream_idx, uncollected_gain_range] = 0.0
                    
                    #Calculating the spread of phases collected from each antennas from regions where we have actual gain values
                    #sigma_phase[antenna_index, stream_idex] = np.std(phase_cals[stream_idx, collected_gain_range], axis = 1)

            delay_residual_map[ant] = residual_delays
            phase_cal_map[ant] = phase_cals
            
        else:
            phase_cal_map[ant] = phase_matrix
            delay_residual_map[ant] = np.zeros(nof_streams,dtype=np.float64)
    
    return delay_residual_map, phase_cal_map #return SNR and sigma_phase if needed
