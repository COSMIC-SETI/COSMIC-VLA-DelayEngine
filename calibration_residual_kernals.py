import numpy as np
import warnings
from scipy.stats import median_abs_deviation as mad

def calc_calibration_ant_grade(ant_to_gains):
    """
    Accept mapping of antenna to gains.
    Returns ant to calibration grade for antenna.

    Args:
        ant_to_gains : A dictionary mapping of antenna name to complex gain matrix of dimension (n_streams, n_chans).
                    {<ant> : [[complex(gains_pol0_tune0)], [complex(gains_pol1_tune0)], [complex(gains_pol0_tune1)], [complex(gains_pol1_tune1)]], ...}

    Return:
        ant_to_grade: A dictionary mapping of antenna name to grade of calibration run.
                    {ant : [[grade, ], ...]}
    """
    ant_to_grade = {}
    for ant, gain_matrix in ant_to_gains.items():
        gain_matrix = np.array(gain_matrix,dtype=np.complex64)
        gain_shape = gain_matrix.shape
        nof_streams = gain_shape[0]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", category=RuntimeWarning)
                ant_to_grade[ant] = np.abs(np.sum(gain_matrix, axis=1))/np.sum(np.abs(gain_matrix),axis=1)
        except (ZeroDivisionError, RuntimeWarning):
            ant_to_grade[ant] = [-1.0]*nof_streams
    return ant_to_grade

def calc_calibration_freq_grade(ant_to_gains):
    """
    Accept mapping of antenna to gains.
    Returns ant to calibration grade for frequency.

    Args:
        ant_to_gains : A dictionary mapping of antenna name to complex gain matrix of dimension (n_streams, n_chans).
                    {<ant> : [[complex(gains_pol0_tune0)], [complex(gains_pol1_tune0)], [complex(gains_pol0_tune1)], [complex(gains_pol1_tune1)]], ...}

    Return:
        freq_to_grade: A matrix of shape (nof_streams, nof_channels)
    """
    antenna = list(ant_to_gains.keys())
    sum_freq = np.zeros(np.array(ant_to_gains[antenna[0]]).shape)
    sum_abs_freq = np.zeros(np.array(ant_to_gains[antenna[0]]).shape)
    freq_to_grade = np.ones(np.array(ant_to_gains[antenna[0]]).shape)
    for _, gain_matrix in ant_to_gains.items():
        gain_matrix = np.array(gain_matrix,dtype=np.complex64)
        gain_shape = gain_matrix.shape
        nof_streams = gain_shape[0]
        sum_freq = sum_freq + gain_matrix
        sum_abs_freq = np.abs(sum_abs_freq) + np.abs(gain_matrix)

    for stream in range(freq_to_grade.shape[0]):
        nonzero_indexes = np.where(sum_abs_freq[stream,:] != 0)[0]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", category=RuntimeWarning)
                freq_to_grade[stream,nonzero_indexes] = np.abs(sum_freq[stream,nonzero_indexes])/sum_abs_freq[stream,nonzero_indexes]
        except (ZeroDivisionError, RuntimeWarning):
            freq_to_grade[stream,nonzero_indexes] = [-1.0]*nof_streams

    return freq_to_grade

def calc_full_grade(ant_to_gains):
    """
    Accept mapping of antenna to gains.
    Returns a single value which is the full sum across all frequencies and antenna.

    Args:
        ant_to_gains : A dictionary mapping of antenna name to complex gain matrix of dimension (n_streams, n_chans).
                    {<ant> : [[complex(gains_pol0_tune0)], [complex(gains_pol1_tune0)], [complex(gains_pol0_tune1)], [complex(gains_pol1_tune1)]], ...}

    Return:
        grade: A single grade value
    """
    antenna = list(ant_to_gains.keys())
    sum_freq = np.zeros(np.array(ant_to_gains[antenna[0]]).shape)
    sum_abs_freq = np.zeros(np.array(ant_to_gains[antenna[0]]).shape)
    for _, gain_matrix in ant_to_gains.items():
        gain_matrix = np.array(gain_matrix,dtype=np.complex64)
        sum_freq = sum_freq + gain_matrix
        sum_abs_freq = np.abs(sum_abs_freq) + np.abs(gain_matrix)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=RuntimeWarning)
            grade = np.abs(np.sum(sum_freq))/np.sum(sum_abs_freq)
    except (ZeroDivisionError, RuntimeWarning):
        grade = -1.0
    return grade

def calc_residuals_from_polyfit(ant_to_gains, observation_frequencies, current_phase_cals, frequency_indices, snr_threshold=4.0):
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
            snr_threshold float : the snr threshold of delay delta to noise above which a calibration run will be deemed suitable for updates to fixed delays/phases to be
                        applied.

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
            nof_streams = phase_matrix.shape[0]
            if ant in ant_to_gains.keys():

                gain_matrix = np.array(ant_to_gains[ant], dtype = np.complex64)
                #Subtract the last applied phases from the gain (incase incorrect)
                current_phase_matrix = np.exp(1j * np.array(phase_matrix))
                new_gain_matrix = gain_matrix * current_phase_matrix

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
                            if np.abs(residual_delay) > 100:
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

def calc_residuals_from_ifft(ant_to_gains, observation_frequencies, current_phase_cals, frequency_indices, sideband, snr_threshold=4.0):
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
        sideband           : A list of integers dictating sideband orientation for the two tunings
        snr_threshold float : the snr threshold of delay delta to noise above which a calibration run will be deemed suitable for updates to fixed delays/phases to be
                        applied.

    Return:
        delay_residual_map : A mapping of antenna to delay residual values in nanoseconds of dimension (n_streams)
                    {<ant> : [residual_delay_pol0_tune0, residual_delay_pol1_tune0, residual_delay_pol0_tune1, residual_delay_pol1_tune1]}, ...}
        phase_cal_map : A mapping of antenna to phase calibration value in radians of dimension (n_streams, n_chans).
                    {<ant> : [[phase_pol0_tune0],[phase_pol1_tune0],
                                        [phase_pol0_tune1],[phase_pol1_tune1]]}, ...} in radians
        snr_map            : A mapping of antenna to the delay peak SNR value
                    {<ant> : [[snr_steam0, snr_stream1...],...]}
        sigma_phase_map    : A mapping of antenna to the standard deviation of the phase calibrations
                    {<ant> : [[sigma_phase_steam0, sigma_phase_stream1...],...]}
    """
    delay_residual_map = {}
    phase_cal_map = {}
    snr_map = {} #the SNR of the ifft
    sigma_phase_map = {} #the spread of the phases
    
    #Finding the number of antennas here. Maybe there is a better way
    # nant = len(ant_to_gains.keys())
    #ant_ind, Also get the antenna index somehow here starting from zero

    for ant, phase_matrix in current_phase_cals.items():
        phase_matrix = np.array(phase_matrix,dtype=float)
        nof_streams = phase_matrix.shape[0]
        if ant in ant_to_gains.keys():
            gain_matrix = np.array(ant_to_gains[ant], dtype = np.complex64)

            #Subtract the last applied phases from the gain (incase incorrect)
            current_phase_matrix = np.exp(1j * np.array(phase_matrix))
            new_gain_matrix = gain_matrix * current_phase_matrix

            nof_tunings,nof_chan = observation_frequencies.shape
            nof_pols = int(nof_streams/nof_tunings)
            ifft_abs_matrix = np.abs(np.fft.ifft(new_gain_matrix, axis = 1))
            max_idxs = np.argmax(ifft_abs_matrix,axis=1)
            ifft_abs_matrix_dB = 10*np.log10(ifft_abs_matrix)
        
            #Adding some lines to do the SNR of the peak here
            ant_sigma = mad(ifft_abs_matrix_dB,  axis=1)
            ant_median = np.median(ifft_abs_matrix_dB, axis=1)

            residual_delays = np.zeros(nof_streams, dtype=np.float64)
            phase_cals = np.zeros(gain_matrix.shape, dtype=np.float64)
            snr = np.zeros(nof_streams, dtype=np.float64)
            sigma_phase = np.zeros(nof_streams, dtype=np.float64)
            
            for tune in range(nof_tunings):
                # deal with sideband by negating frequencies
                obsfreqs = sideband[tune]*observation_frequencies[tune,:]
                #find range outside collected gains
                uncollected_gain_range = np.setdiff1d(np.arange(obsfreqs.size), frequency_indices[tune], assume_unique = True)
                chan_width = obsfreqs[1] - obsfreqs[0]
                freqs = np.fft.fftfreq(nof_chan, chan_width)
                tlags = freqs*1e9
                for pol in range(nof_pols):   #probably unecessary - could do with some matrix mult stuff
                    #some binary logic
                    stream_idx = int(str(tune)+str(pol),2)
                    
                    #Calculating the SNR in dB
                    snr[stream_idx] = (ifft_abs_matrix_dB[stream_idx, max_idxs[stream_idx]] - ant_median[stream_idx])/ant_sigma[stream_idx]

                    residual_delay = -1.0 * tlags[max_idxs[stream_idx]]
                    #Now rather than using the abs(residual_delay) value, use
                    if snr[stream_idx] > snr_threshold:
                        residual_delays[stream_idx] = residual_delay
                        gain_from_residual_delay = np.exp(2j*np.pi*(obsfreqs*1e-9)*residual_delays[stream_idx])
                        phase_cals[stream_idx] = np.angle(new_gain_matrix[stream_idx,:]/gain_from_residual_delay)
                        #zero all phases outside the collected gains range
                        phase_cals[stream_idx, uncollected_gain_range] = 0.0
                    else:
                        residual_delays[stream_idx] = 0.0
                        phase_cals[stream_idx] = phase_matrix[stream_idx]
                    
                    #Calculating the spread of phases collected from each antennas from regions where we have actual gain values
                    sigma_phase[stream_idx] = np.std(phase_cals[stream_idx, frequency_indices[tune]])

            delay_residual_map[ant] = residual_delays
            phase_cal_map[ant] = phase_cals
            snr_map[ant] = snr
            sigma_phase_map[ant] = sigma_phase
            
        else:
            phase_cal_map[ant] = phase_matrix
            delay_residual_map[ant] = np.zeros(nof_streams,dtype=np.float64)
    
    return delay_residual_map, phase_cal_map, snr_map, sigma_phase_map
