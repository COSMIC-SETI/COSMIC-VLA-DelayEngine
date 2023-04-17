import numpy as np
import json
import os
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
        ant_to_grade[ant] = np.abs(np.sum(gain_matrix, axis=1))/np.sum(np.abs(gain_matrix),axis=1)
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
        sum_freq = sum_freq + gain_matrix
        sum_abs_freq = np.abs(sum_abs_freq) + np.abs(gain_matrix)

    for stream in range(freq_to_grade.shape[0]):
        nonzero_indexes = np.where(sum_abs_freq[stream,:] != 0)[0]
        freq_to_grade[stream,nonzero_indexes] = np.abs(sum_freq[stream,nonzero_indexes])/sum_abs_freq[stream,nonzero_indexes]

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

    grade = np.abs(np.sum(sum_freq))/np.sum(sum_abs_freq)
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

def calc_residuals_from_ifft(ant_to_gains, observation_frequencies, current_phase_cals, frequency_indices, snr_threshold=4.0, output_dir=None):
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
        snr_threshold float : the snr threshold of delay delta to noise above which a calibration run will be deemed suitable for updates to fixed delays/phases to be
                        applied.
        output_dir : an output directory for intermediate products within the kernel to be saved to. Only if provided, are these products are saved.

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

    if output_dir is not None:
        ifft_map = {}
        residual_delay_gain_map = {}
    
    #Finding the number of antennas here. Maybe there is a better way
    # nant = len(ant_to_gains.keys())
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

            if output_dir is not None:
                ifft_map[ant] = ifft_abs_matrix.tolist()
            
            #Adding some lines to do the SNR of the peak here
            ant_sigma = mad(ifft_abs_matrix,  axis=1)
            ant_median = np.median(ifft_abs_matrix, axis=1)

            residual_delays = np.zeros(nof_streams, dtype=np.float64)
            phase_cals = np.zeros(gain_matrix.shape, dtype=np.float64)
            snr = np.zeros(nof_streams, dtype=np.float64)
            sigma_phase = np.zeros(nof_streams, dtype=np.float64)
            if output_dir is not None:
                residual_delay_gain = np.zeros(gain_matrix.shape, dtype=complex)
            
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
                    snr[stream_idx] = (ifft_abs_matrix[stream_idx, max_idxs[stream_idx]] - ant_median[stream_idx])/ant_sigma[stream_idx]

                    residual_delay = -1.0 * tlags[max_idxs[stream_idx]]
                    #Now rather than using the abs(residual_delay) value, use
                    if snr[stream_idx] > snr_threshold:
                        residual_delays[stream_idx] = residual_delay
                        gain_from_residual_delay = np.exp(2j*np.pi*(observation_frequencies[tune,:]*1e-9)*residual_delays[stream_idx])
                        if output_dir is not None:
                            residual_delay_gain[stream_idx,:] = gain_from_residual_delay
                        phase_cals[stream_idx] = np.angle(new_gain_matrix[stream_idx,:]/gain_from_residual_delay)
                        #zero all phases outside the collected gains range
                        phase_cals[stream_idx, uncollected_gain_range] = 0.0
                    else:
                        residual_delays[stream_idx] = 0.0
                        phase_cals[stream_idx] = phase_matrix[stream_idx]
                    
                    #Calculating the spread of phases collected from each antennas from regions where we have actual gain values
                    sigma_phase[stream_idx] = np.std(phase_cals[stream_idx, frequency_indices[tune]])
            if output_dir is not None:
                residual_delay_gain_map[ant+"_real"] = residual_delay_gain.real.tolist()
                residual_delay_gain_map[ant+"_imag"] = residual_delay_gain.imag.tolist()
            delay_residual_map[ant] = residual_delays
            phase_cal_map[ant] = phase_cals
            snr_map[ant] = snr
            sigma_phase_map[ant] = sigma_phase
            
        else:
            phase_cal_map[ant] = phase_matrix
            delay_residual_map[ant] = np.zeros(nof_streams,dtype=np.float64)
    
    if output_dir is not None:
        with open(os.path.join(output_dir,"delay_residual_gain.json"), 'w') as f:
            json.dump(residual_delay_gain_map, f)
        with open(os.path.join(output_dir,"ifft_map.json"), 'w') as f:
            json.dump(ifft_map, f)
        np.save(os.path.join(output_dir,"fft_freqs.npy"),freqs)
    
    return delay_residual_map, phase_cal_map, snr_map, sigma_phase_map
