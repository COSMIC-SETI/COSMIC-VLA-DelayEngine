import sys, os
import time
from blimpy import GuppiRaw
from cosmic.redis_actions import redis_obj, redis_hget_keyvalues
from delay_snapshot import gen_delay_vectors
from delaymodel import DelayModel
from astropy.coordinates import SkyCoord
import numpy as np
import matplotlib
import tqdm as tq
from matplotlib import pyplot as plt
import argparse

def uvh5_block_to_matrix(file : str):
    """
    Extract a floating point complex matrix from a uvh5 file.
    Args:
        file1: path to raw file 1 
    Returns:
        a matrix of dimension (baseline, nchan, time, pols)
    """

def raw_block_to_matrix(file1 : str, file2 : str):
    """
    Extract 2 floating point complex matrices from the raw files provided and 
    return them.
    Args:
        file1: path to raw file 1 
        file2: path to raw file 2 

    Returns:
        2 matrices of dimension (nant, nchan, time, pols)
    """
    guppi_obj_1 = GuppiRaw(file1) #Instantiating guppi object 1
    header1 = guppi_obj_1.read_first_header() # Reading the first header
    nant1 = int(header1['NANTS'])
    nant_chans1 = int(header1['OBSNCHAN'])
    nchan1 = int(nant_chans1/nant1)
    npols1 = int(header1['NPOL'])
    nbits1 = int(header1['NBITS'])
    blocksize1 = header1['BLOCSIZE']
    ntsamp_block1 = int(blocksize1/(nant_chans1*npols1*2*(nbits1/8))) # Number of time samples in the block
    guppi_obj_2 = GuppiRaw(file2) #Instantiating guppi object 2
    header2 = guppi_obj_2.read_first_header() # Reading the second header 
    nant2 = int(header2['NANTS'])
    nant_chans2 = int(header2['OBSNCHAN'])
    nchan2 = int(nant_chans2/nant2)
    npols2 = int(header2['NPOL'])
    nbits2 = int(header2['NBITS'])
    blocksize2 = header2['BLOCSIZE']
    ntsamp_block2 = int(blocksize2/(nant_chans2*npols2*2*(nbits2/8))) # Number of time samples in the block

    #we only need to read 1 block to compute the delay and delay rate
    data1 = np.zeros((nant1, nchan1, ntsamp_block1, npols1), dtype = 'complex64') 
    data2 = np.zeros((nant2, nchan2, ntsamp_block2, npols2), dtype = 'complex64') 
    head_block1, data_block1 = guppi_obj_1.read_next_data_block()
    head_block2, data_block2 = guppi_obj_2.read_next_data_block()

    data1[:, :, :, :] = data_block1.reshape(nant1, nchan1, ntsamp_block1, npols1)
    data2[:, :, :, :] = data_block2.reshape(nant2, nchan2, ntsamp_block2, npols2)

    return data1, head_block1, data2, head_block2

def fetch_delays_for_obsinfo(ra : float, dec : float, time : float, duration : float, ntime : int, ants : list):
    """
    Run the delay model for the provided observation information and provide (in conjunction
    with the calibration delays) the expected delay difference for all baselines in a dict.
    
    Args:
        ra          : floating point value (in degrees)
        dec         : floatingg point value (in degrees)
        time        : the time of the observation in unix time (seconds)
        duration    : the observation time duration (seconds)
        ntime       : the number of time samples to compute over
        ants        : a list of antennas to collect delays for
        """
    time_range = np.linspace(time, time + duration, ntime)
    calib_delays = redis_hget_keyvalues(redis_obj, "META_calibrationDelays", keys=ants)
    delay_model = DelayModel(redis_obj)
    delay_model.source = SkyCoord(ra, dec, unit='deg')
    model_delays = delay_model.calculate_delay(publish = False)

    return gen_delay_vectors(model_delays, calib_delays, time_range, use_calib=True)

def calc_expected_delays(ant_to_delay_dict, ant_to_delay_rate_dict, baseline_type : str = "cross-ant"):
    """
    Accepts an antenna to delay dictionay and antenna to delay rate dictionary and
    returns a baseline to expected delay map and a baseline to expected delay rate
    map.

    Args:
        ant_to_delay_dict : a dictionary of antenna name to delay to be applied 
        ant_to_delay_rate_dict : a dictionary of antenna name to delay_rate to be applied
        baseline_type : this string indicates which baseline type to compute the delay difference for.
                        Options are: 
                            - `cross-ant` (ant_a (pol_x, tuning_0) to ant_b (pol_x, tuning_0) etc).
                                This will compute the baseline delay difference between antenna for a single 
                                polarisation and tuning.
                                Returns dict{baseline_name: array[delay_diff(xx)], array[delay_diff(yy)]}

                            - `cross-pol` (ant_a (pol_x, tuning_0) to ant_a (pol_y, tuning_0) etc).
                                This will compute the baseline delay difference between polarisations x & y
                                for a single antenna and tuning.
                                Returns dict{antname: array[delay_diff_tune_0(x,y)], array[delay_diff_tune_1(x,y)]}
                    
                            - `cross-tuning` (ant_a (pol_x, tuning_0) to ant_a (pol_x, tuning_1) etc).
                                This will compute the baseline delay difference between tunings 0 & 1 
                                for a single antenna and polarisation. 
                                Returns dict{antname: array[delay_diff_x(tune_0, tune_1)], array[delay_diff_y(tune_0, tune_1)]}
    """

    if baseline_type == "cross-pol":
        ant_to_pol_delay_diff = {}
        for ant, delay in ant_to_delay_dict.items():
            ant_to_pol_delay_diff[ant] = (delay[1,:] - delay[0,:], delay[3,:] - delay[2,:]) #tuple cross corr pol
        return ant_to_pol_delay_diff

    elif baseline_type == "cross-tuning":
        ant_to_tuning_delay_diff = {}
        for ant, delay in ant_to_delay_dict.items():
            ant_to_tuning_delay_diff[ant] = (delay[2,:] - delay[0,:], delay[3,:] - delay[1,:]) #tuple diff across tuning
        return ant_to_tuning_delay_diff
    
    elif baseline_type =="cross-ant":
        baseline_delay_diff = {}
        baseline_delay_rate_diff = {}

        nant = len(ant_to_delay_dict.keys())

        for ant1,delay1 in ant_to_delay_dict.items():
            for ant2,delay2 in ant_to_delay_dict.items():
                if ant2 != ant1:
                    baseline_delay_diff[f"{ant1}-{ant2}"] = (delay1[0,:] - delay2[0,:], delay1[1,:] - delay2[1,:],
                                                         delay1[2,:] - delay2[2,:], delay1[3,:] - delay2[3,:]) 
        for ant1,delay1 in ant_to_delay_rate_dict.items():
            for ant2,delay2 in ant_to_delay_rate_dict.items():
                if ant2 != ant1:
                    baseline_delay_rate_diff[f"{ant1}-{ant2}"] = (delay1[0,:] - delay2[0,:], delay1[1,:] - delay2[1,:],
                                                         delay1[2,:] - delay2[2,:], delay1[3,:] - delay2[3,:]) 

        return baseline_delay_diff, baseline_delay_rate_diff

def get_cross_pol_delays(block_matrix, tbin, corr=True):
    """
    For a single input matrix of dimension (nant, nchan, time, pols), return a matrix of dimension (nant, time).
    Use the tbin value provided to calculate the delay corresponding to the time index calculated.

    Args:
        block_matrix    : matrix of dimension (nant, nchan, time, pols)
        tbin            : float value in seconds
        corr            : whether or not correlation is required
    """
    if corr:
        corr = (np.conj(block_matrix[:, :, :, 0]) *(block_matrix[:, :, :, 1])) #cross pol correlations -> dims=(nant, nchan, ntime)
    else:
        corr = block_matrix

    padding = int(2**14)
    nant,_,ntime = corr.shape
    delay_indx = np.zeros((nant, ntime))

    for ant in range(nant):
        corr_ifft = np.fft.ifft(corr[ant,:,:], n=padding, axis=0)
        delay_indx[ant, :] = corr_ifft.argmax(axis=0)
    
    tbin_fine = (tbin/(padding)) * 1e9 # ns
    
    return delay_indx*tbin_fine

def get_cross_tuning_delays(block_matrix1, block_matrix2, tbin):
    """
    For two input matrices of dimension (nant, nchan, time, pols), return two matrices of dimension (nant, time).
    One for each polarisation.
    Use the tbin value provided to calculate the delay corresponding to the time index calculated.
    Note that this operation is only usable in the case of RAW file analysis since the correlation pipeline does
    not perform cross-tuning correlation in UVH5.

    Args:
        block_matrix1    : matrix of dimension (nant, nchan, time, pols)
        block_matrix2    : matrix of dimension (nant, nchan, time, pols)
        tbin             : float value in seconds
    """
    padding = int(2**14)

    corr_x = (np.conj(block_matrix1[:, :, :, 0]) *(block_matrix2[:, :, :, 0])) #cross tuning correlation for x -> dims=(nant, nchan, ntime)
    corr_y = (np.conj(block_matrix1[:, :, :, 1]) *(block_matrix2[:, :, :, 1])) #cross tuning correlation for y -> dims=(nant, nchan, ntime)

    nant,nchan,ntime = corr_x.shape
    delay_indx_x = np.zeros((nant, ntime))
    delay_indx_y = np.zeros((nant, ntime))
    for ant in range(nant):
        corrx_ifft = np.fft.ifft(corr_x[ant,:,:], n=padding, axis=0)
        corry_ifft = np.fft.ifft(corr_y[ant,:,:], n=padding, axis=0)
        delay_indx_x[ant, :] = corrx_ifft.argmax(axis=0)
        delay_indx_y[ant, :] = corry_ifft.argmax(axis=0)
    
    tbin_fine = tbin/(padding) * 1e9 # ns
    
    return delay_indx_x * tbin_fine, delay_indx_y * tbin_fine

def get_cross_ant_delays(block_matrix, tbin, corr=True):
    """
    For input matrix of dimension (nant, nchan, time, pols), return a matrix of dimension (ncorr, time, npol).
    Use the tbin value provided to calculate the delay corresponding to the time index calculated.

    Args:
        block_matrix1    : matrix of dimension (nant, nchan, time, pols)
        tbin             : float value in seconds
        corr             : whether or not correlation is required
    """
    nant, nchan, ntime, npols = block_matrix.shape
    n_baselines = (nant**2 - nant)/2
    padding = int(2**14)

    if corr:
        baseline_corr = np.zeros((n_baselines, nchan, ntime, npols))
        j = 0
        i = 0
        for b in range(n_baselines):
            j+=1
            if j==nant-1:
                i += 1
                j = i+1    
            baseline_corr[b,:,:,:] = (np.conj(block_matrix[j, :, :, :]) * (block_matrix[i, :, :, :]))
    else:
        baseline_corr = block_matrix
    
    baseline_delays = np.zeros((n_baselines, ntime))
    for b in range(n_baselines):
        corr_base_ifft = np.fft.ifft(baseline_delays[b, :, :, :], n = padding, axis=0)
        baseline_delays[b,:] = corr_base_ifft.argmax(axis=0) 
        
    tbin_fine = (tbin/padding) * 1e9 # ns
    return baseline_delays * tbin_fine

def find_delay_index(vec1, vec2, timeslice=1):
    """
    Return all delay indexes in a dictionary
    """
    zero_pad = np.zeros(int(2**13),dtype='complex64')
    t_corr1 = (np.conj(vec1[:, :, timeslice, 0]) *(vec1[:, :, timeslice, 1])).flatten() #file one, cross pol correlations
    t_corr2 = (np.conj(vec2[:, :, timeslice, 0]) *(vec2[:, :, timeslice, 1])).flatten() #file two, cross pol correlations
    t_corr3 = (np.conj(vec1[:, :, timeslice, 0]) *(vec2[:, :, timeslice, 0])).flatten() #file one, file two, first pol, cross tuning correlations
    t_corr4 = (np.conj(vec1[:, :, timeslice, 1]) *(vec2[:, :, timeslice, 1])).flatten() #file one, file two, second pol, cross tuning correlations
    t_corr_ifft_1 = np.abs(np.fft.ifft(np.hstack((zero_pad, t_corr1, zero_pad))))
    t_corr_ifft_2 = np.abs(np.fft.ifft(np.hstack((zero_pad, t_corr2, zero_pad))))
    t_corr_ifft_3 = np.abs(np.fft.ifft(np.hstack((zero_pad, t_corr3, zero_pad))))
    t_corr_ifft_4 = np.abs(np.fft.ifft(np.hstack((zero_pad, t_corr4, zero_pad))))

    # plt.plot(np.angle(t_corr1))
    # plt.savefig("tmp11.png")
    # plt.close()
    # plt.plot(np.angle(t_corr2))
    # plt.savefig("tmp12.png")
    # plt.close()
    # plt.plot(np.angle(t_corr3))
    # plt.savefig("tmp13.png")
    # plt.close()
    # plt.plot(np.angle(t_corr4))
    # plt.savefig("tmp14.png")
    # plt.close()

    t_bin = 1
    tot_time = t_bin * 1024
    t_fine = tot_time / (2**14 + 1024)
    x_axis = np.linspace(0, tot_time, num = 2**14 + 1024)

    delay_index_1 = int(np.where(t_corr_ifft_1 == np.max(t_corr_ifft_1))[0])
    delay_index_2 = int(np.where(t_corr_ifft_2 == np.max(t_corr_ifft_2))[0])
    delay_index_3 = int(np.where(t_corr_ifft_3 == np.max(t_corr_ifft_3))[0])
    delay_index_4 = int(np.where(t_corr_ifft_4 == np.max(t_corr_ifft_4))[0])

    # plt.plot(x_axis, t_corr_ifft_1, color='k')
    # plt.title(f'Measured time delay for tuning 0 cross-pol correlation:\n{delay_index_1*t_fine}us')
    # plt.xlabel("time bins (us)")
    # plt.savefig("tmp1.png")
    # plt.close()
    # plt.plot(x_axis, t_corr_ifft_2, color='k')
    # plt.title(f'Measured time delay for tuning 1 cross-pol correlation:\n{delay_index_2*t_fine}us')
    # plt.savefig("tmp2.png")
    # plt.xlabel("time bins (us)")    
    # plt.close()
    # plt.plot(x_axis, t_corr_ifft_3, color='k')
    # plt.title(f'Measured time delay for pol 0 cross-tuning correlation:\n{delay_index_3*t_fine}us')
    # plt.savefig("tmp3.png")
    # plt.xlabel("time bins (us)")
    # plt.close()
    # plt.plot(x_axis, t_corr_ifft_4, color='k')
    # plt.title(f'Measured time delay for pol 1 cross-tuning correlation:\n{delay_index_4*t_fine}us')
    # plt.savefig("tmp4.png")
    # plt.xlabel("time bins (us)")
    # plt.close()

    print(delay_index_1)
    print(delay_index_2)
    print(delay_index_3)
    print(delay_index_4)
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="For a given set of RAW files compute the effective time delay"
    )
    parser.add_argument(
        "--file1",
        type=str,
        help="Raw/UVH5 file 1 to load"
    )
    parser.add_argument(
        "--file2",
        type=str,
        help="Raw/UVH5 file 2 to load"
    )
    parser.add_argument(
        "-p",
        "--plot",
        action='store_true',
        help="Generate plots for all expected delays. If ignored, print save result to csv."
    )
    args = parser.parse_args()
    data1, head1, data2, head2 = raw_block_to_matrix(args.file1, args.file2)
    ra_deg = head1["RA_STR"]
    dec_deg = head1["DEC_STR"]
    tbin = head1["TBIN"]
    ntime =  (head1["BLOCSIZE"]*8)/(head1["OBSNCHAN"]*head1["NPOL"]*2*head1["NBITS"])
    tstart = head1["SYNCTIME"] + head1["PKTIDX"] * head1["TBIN"] * ntime/head1["PIPERBLK"]
    ants = head1["ANTNMS00"]
    t_pol_delays= get_cross_pol_delays(data2, tbin)
    ntime = len(t_pol_delays[0])
    ant_to_delay_dict, ant_to_delay_rate_dict = fetch_delays_for_obsinfo(ra_deg, dec_deg, tstart, 1, ntime, ants)
    delays = calc_expected_delays(ant_to_delay_dict, ant_to_delay_rate_dict, baseline_type="cross-pol")
    
    