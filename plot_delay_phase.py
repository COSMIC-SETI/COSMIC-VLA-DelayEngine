"""
Written by Savin Shynu Varghese and 
edited by Talon Myburgh.
22/01/2023
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse

def plot_gain_grade(ant_to_grade, outdir=None, outfilestem=None,
                    source_name=None):
    """
    Function that accepts a dictionary of antenna to calibration grade and plots the results.

    Args:
        ant_to_grade         : a dictionary mapping of {ant: [grade, 1]}
        outdir               : output directory to which to save the plots
        outfilestem          : string to prefix to the plot filename
        source_name          : string name of source to use in plot title
    
    Returns:
        grade_file_path      : str filepath to the snr vs ant and sigma vs ant png plot    
    """

    if outdir is not None and not os.path.exists(outdir):
        os.makedirs(outdir)

    #Getting the antenna info from the keys
    antennas = list(ant_to_grade.keys())

    #plotting the snr vs antennas and sigma spread - can assume they contain same ant values as they are
    #derived at the same point
    i = 0
    fig, ax = plt.subplots(1, 1, sharex = True, constrained_layout=True, figsize = (10,8))
    for ant, grade in ant_to_grade.items():
        ax.plot(i, grade[0], '.', color='royalblue')
        ax.plot(i, grade[1], '.', color='crimson')
        ax.plot(i, grade[2], '.', color='orange')
        ax.plot(i, grade[3], '.', color='forestgreen')
        i=i+1

    ax.legend(["AC0","AC1","BD0","BD1"], loc = 'upper right')
    ax.set_ylabel("abs(sum(gains))/sum(abs(gains))")
    ax.set_xticks(np.arange(len(antennas)))
    ax.set_xticklabels(antennas)
    fig.suptitle(f"""
    Calculated Gain grade from
    {outfilestem}
    for source {source_name}""")
    fig.supxlabel("Antennas")

    if outfilestem is not None:
        outfile_name = outfilestem+"grade_vs_ant.png"
    else:
        outfile_name = "grade_vs_ant.png"

    if outdir is not None:
        grade_file_path = os.path.join(outdir, outfile_name)
    else:
        grade_file_path = outfile_name
         
    plt.savefig(grade_file_path, dpi = 150)
    plt.close()
    return grade_file_path

def plot_snr_and_phase_spread(ant_to_snr, ant_to_sigma_phase, outdir=None, outfilestem=None,
                    source_name=None):
    """
    Function that accepts two dictionaries. One being the antenna to ifft SNR of delay-peak to noise,
    and the other being the antenna to std_deviation of the phase.

    Args:
        ant_to_snr           : a dictionary mapping of {ant: [snr, 1]}
        ant_to_sigma_phase   : a dictionary mapping of {ant: [std_deviation, 1]} 
        outdir               : output directory to which to save the plots
        outfilestem          : string to prefix to the plot filename
        source_name          : string name of source to use in plot title
    Returns:
        snr_and_sigma_file_path      : str filepath to the snr vs ant and sigma vs ant png plot
    """

    if outdir is not None and not os.path.exists(outdir):
        os.makedirs(outdir)

    #Getting the antenna info from the keys
    antennas = list(ant_to_snr.keys())

    #plotting the snr vs antennas and sigma spread - can assume they contain same ant values as they are
    #derived at the same point
    i = 0
    fig, ax = plt.subplots(2, 1, sharex = True, constrained_layout=True, figsize = (10,12))
    for ant, snr in ant_to_snr.items():
        ax[0].plot(i, snr[0], '.', color='royalblue')
        ax[0].plot(i, snr[1], '.', color='crimson')
        ax[0].plot(i, snr[2], '.', color='orange')
        ax[0].plot(i, snr[3], '.', color='forestgreen')

        ax[1].plot(i, ant_to_sigma_phase[ant][0], '.', color='royalblue')
        ax[1].plot(i, ant_to_sigma_phase[ant][1], '.', color='crimson')
        ax[1].plot(i, ant_to_sigma_phase[ant][2], '.', color='orange')
        ax[1].plot(i, ant_to_sigma_phase[ant][3], '.', color='forestgreen')
        i=i+1

    ax[0].set_title("IFFT delay peak SNR")
    ax[1].set_title("Standard deviation of phase calibration values")

    ax[0].legend(["AC0","AC1","BD0","BD1"], loc = 'upper right')
    ax[0].set_ylabel("SNR")
    ax[1].legend(["AC0","AC1","BD0","BD1"],loc = 'upper right')
    ax[1].set_ylabel("Std deviation of phase cals")
    ax[0].set_xticks(np.arange(len(antennas)))
    ax[0].set_xticklabels(antennas)
    ax[1].set_xticks(np.arange(len(antennas)))
    ax[1].set_xticklabels(antennas)
    fig.suptitle(f"""
    Calculated SNR and std deviation from
    {outfilestem}
    for source {source_name}""")
    fig.supxlabel("Antennas")

    if outfilestem is not None:
        outfile_name = outfilestem+"snr_and_sigma_vs_ant.png"
    else:
        outfile_name = "snr_and_sigma_vs_ant.png"

    if outdir is not None:
        snr_and_sigma_file_path = os.path.join(outdir, outfile_name)
    else:
        snr_and_sigma_file_path = outfile_name
         
    plt.savefig(snr_and_sigma_file_path, dpi = 150)
    plt.close()
    return snr_and_sigma_file_path
    
def plot_gain_phase(ant_to_gains, observation_frequencies, frequency_indices, fit_method="linear", outdir=None, outfilestem=None, source_name=None):
    """
    Plot the phase of the received complex gain values per antenna. In the event that
    the fitting method is "linear", unwrap the phases, before plotting.

    Args:
        ant_to_gains : {<ant> : [[complex(gains_pol0_tune0)], [complex(gains_pol1_tune0)], [complex(gains_pol0_tune1)], [complex(gains_pol1_tune1)]], ...}
        observation_frequencies : list of dimension (n_tunings, nchans) in Hz
        frequency_indices : frequency indices that decide which parts of the plot to make color
        fit_method : str indicating the fit method used in the calibration run
        outdir : str directory path to save the plots to
        outfilestem : str filename stem to use
        source_name : str name of source to use in plot title
    Returns:
        phase_file_path_ac : str filepath to the phase vs freq plot for ac tuning
        phase_file_path_bd : str filepath to the phase vs freq plot for bd tuning
    """
    if outdir is not None and not os.path.exists(outdir):
        os.makedirs(outdir)

    #Getting the antenna info from the keys
    antennas = list(ant_to_gains.keys())

    #plotting the phases vs antennas
    #make a grid plot of phase vs freq for all antennas
    grid_x = 6
    grid_y = 5
    fig, axs = plt.subplots(grid_x, grid_y, sharex  = True, sharey = True, constrained_layout=True, figsize = (12,14))

    #Tuning 0
    uncollected_gains = np.delete(np.arange(observation_frequencies[0,:].size), frequency_indices[0])
    for i in range(grid_x):
        for j in range(grid_y):
            ant_ind = (i*grid_y)+j
            if ant_ind < len(antennas):
                phases_pol0 = np.angle(ant_to_gains[antennas[ant_ind]][0]) * (180.0/np.pi)
                phases_pol1 = np.angle(ant_to_gains[antennas[ant_ind]][1]) * (180.0/np.pi)
                if fit_method=="linear":
                    phases_pol0 = np.unwrap(phases_pol0)
                    phases_pol1 = np.unwrap(phases_pol1)
                axs[i,j].plot(
                    observation_frequencies[0,frequency_indices[0]], phases_pol0[frequency_indices[0]],
                      '.',  label = "AC0"
                    )
                axs[i,j].plot(
                    observation_frequencies[0,uncollected_gains], phases_pol0[uncollected_gains],
                      '.', color='grey'
                      )
                axs[i,j].plot(
                    observation_frequencies[0,frequency_indices[0]], phases_pol1[frequency_indices[0]],
                      '.',  label = "AC1"
                      )
                axs[i,j].plot(
                    observation_frequencies[0,uncollected_gains], phases_pol1[uncollected_gains],
                      '.', color='grey'
                      )
                axs[i,j].set_title(f"{antennas[ant_ind]}_AC")
                axs[i,j].legend(loc = 'upper right')

    fig.suptitle(f"Recorded Phase vs Freq from\n {outfilestem}\nfor source {source_name} and  tuning AC")
    fig.supylabel("Phase (degrees)")
    fig.supxlabel("Frequency Channels")

    if outfilestem is not None:
        outfile_name = outfilestem+"recorded_phaseAC_vs_antennas.png"
    else:
        outfile_name = "recorded_phaseAC_vs_antennas.png"

    if outdir is not None:
        phase_file_path_ac = os.path.join(outdir, outfile_name)
    else:
        phase_file_path_ac = outfile_name
         
    plt.savefig(phase_file_path_ac, dpi = 150)
    plt.close() 

    fig, axs = plt.subplots(grid_x, grid_y, sharex  = True, sharey = True, constrained_layout=True, figsize = (12,14))

    #Tuning 1
    uncollected_gains = np.delete(np.arange(observation_frequencies[1,:].size), frequency_indices[1])
    for i in range(grid_x):
        for j in range(grid_y):
            ant_ind = (i*grid_y)+j
            if ant_ind < len(antennas):
                phases_pol0 = np.angle(ant_to_gains[antennas[ant_ind]][2]) * (180.0/np.pi)
                phases_pol1 = np.angle(ant_to_gains[antennas[ant_ind]][3]) * (180.0/np.pi)
                if fit_method=="linear":
                    phases_pol0 = np.unwrap(phases_pol0)
                    phases_pol1 = np.unwrap(phases_pol1)

                axs[i,j].plot(
                    observation_frequencies[1,frequency_indices[1]], phases_pol0[frequency_indices[1]],
                      '.',  label = "BD0"
                    )
                axs[i,j].plot(
                    observation_frequencies[1,uncollected_gains], phases_pol0[uncollected_gains],
                      '.', color='grey'
                      )
                axs[i,j].plot(
                    observation_frequencies[1,frequency_indices[1]], phases_pol1[frequency_indices[1]],
                      '.',  label = "BD1"
                      )
                axs[i,j].plot(
                    observation_frequencies[1,uncollected_gains], phases_pol1[uncollected_gains],
                      '.', color='grey'
                      )
                axs[i,j].set_title(f"{antennas[ant_ind]}_BD")
                axs[i,j].legend(loc = 'upper right')

    fig.suptitle(f"Recorded Phase vs Freq from\n {outfilestem}\nfor source {source_name} and tuning BD")
    fig.supylabel("Phase (degrees)")
    fig.supxlabel("Frequency Channels")

    if outfilestem is not None:
        outfile_name = outfilestem+"recorded_phaseBD_vs_antennas.png"
    else:
        outfile_name = "recorded_phaseBD_vs_antennas.png"

    if outdir is not None:
        phase_file_path_bd = os.path.join(outdir, outfile_name)
    else:
        phase_file_path_bd = outfile_name
         
    plt.savefig(phase_file_path_bd, dpi = 150)
    plt.close() 

    return phase_file_path_ac, phase_file_path_bd

def plot_delay_phase(residual_delays_dict, phase_dict, frequency_matrix, outdir=None, outfilestem=None,
                    source_name=None):
    """
    Function to collect the delay and phase information
    from the residual files generated after the calibration step
    and plot them.

    Args:
        residual_delays_dict : a dictionary mapping of {ant: [nof_streams, 1]}
        phase_dict           : a dictionary mapping of {ant: [nof_streams, nof_frequencies]} 
        frequency_matrix     : full observation channel frequencies dims(nof_tunings, nof_frequencies)
        outdir               : output directory to which to save the plots
        outfilestem          : string to prefix to the plot filename
        source_name          : string name of source to use in plot title
    Returns:
        delay_file_path : str filepath to the residual delay vs ant png plot
        phase_file_path_ac : str filepath to the phase vs freq plot for ac tuning
        phase_file_path_bd : str filepath to the phase vs freq plot for bd tuning
    """
    if outdir is not None and not os.path.exists(outdir):
        os.makedirs(outdir)

    #Getting the antenna info from the keys
    antennas = list(residual_delays_dict.keys())

    if antennas is not None:
        test_delay = residual_delays_dict[antennas[0]]
        test_phase = phase_dict[antennas[0]]
    else:
        print("No antenna key in the files")
        return

    delay_shape = tuple([len(antennas),])+np.squeeze(test_delay).shape
    phase_shape = tuple([len(antennas),])+np.squeeze(test_phase).shape
    
    #Saving the delay and phase value into an array for easy plotting
    delay_dat = np.zeros(delay_shape)
    phase_dat = np.zeros(phase_shape)

    # Grabing the values
    for i, ant in enumerate(antennas):
        delay_dat[i,:] = residual_delays_dict[ant]
        phase_dat[i,...] = phase_dict[ant]
    
    #wrap phases:
    phase_dat = (phase_dat + np.pi)%(2*np.pi) - np.pi
    #converting the phase to degrees
    phase_dat *= (180.0/np.pi)
    
    #plotting the residual delays vs antennas
    fig, ax = plt.subplots(2, 1, sharex = True, constrained_layout=True, figsize = (10,12))

    for i in range(2):
        ax[i].plot(delay_dat[:,0], '.',  label = "AC0")
        ax[i].plot(delay_dat[:,1], '.',  label = "AC1")
        ax[i].plot(delay_dat[:,2], '.',  label = "BD0")
        ax[i].plot(delay_dat[:,3], '.',  label = "BD1")

        ax[i].legend(loc = 'upper right')
        ax[i].set_xticks(np.arange(len(antennas)))
        ax[i].set_xticklabels(antennas)
    
    ax[0].set_title("Residual delay vs antenna")
    ax[1].set_title("Residual delay vs antenna - Zoomed [-5ns, 5ns)")
    ax[1].set_ylim(-5, 5)
    
    fig.suptitle(f"""
    Calculated Residual delay from
    {outfilestem}
    for source {source_name}""")
    fig.supylabel("Residual Delays (ns)")
    fig.supxlabel("Antennas")

    if outfilestem is not None:
        outfile_name = outfilestem+"delays_vs_ant.png"
    else:
        outfile_name = "delays_vs_ant.png"

    if outdir is not None:
        delay_file_path = os.path.join(outdir, outfile_name)
    else:
        delay_file_path = outfile_name
         
    plt.savefig(delay_file_path, dpi = 150)
    plt.close()
    

    #make a grid plot of phase vs freq for all antennas
    grid_x = 6
    grid_y = 5

    fig, axs = plt.subplots(grid_x, grid_y, sharex  = True, sharey = True, constrained_layout=True, figsize = (12,14))
    for i in range(grid_x):
        for j in range(grid_y):
            ant_ind = (i*grid_y)+j
            if ant_ind < len(antennas):
                axs[i,j].plot(frequency_matrix[0,:], phase_dat[ant_ind,0,:], '.',  label = "AC0")
                axs[i,j].plot(frequency_matrix[0,:], phase_dat[ant_ind,1,:], '.',  label = "AC1")
                axs[i,j].set_title(f"{antennas[ant_ind]}_AC")
                axs[i,j].legend(loc = 'upper right')
    
    fig.suptitle(f"Calculated Phase Calibration Coefficients vs Freq from\n{outfilestem}\nfor source {source_name} and tuning AC")
    fig.supylabel("Phase (degrees)")
    fig.supxlabel("Frequency Channels")

    if outfilestem is not None:
        outfile_name = outfilestem+"phaseAC_vs_antennas.png"
    else:
        outfile_name = "phaseAC_vs_antennas.png"

    if outdir is not None:
        phase_file_path_ac = os.path.join(outdir, outfile_name)
    else:
        phase_file_path_ac = outfile_name
         
    plt.savefig(phase_file_path_ac, dpi = 150)
    plt.close() 
    
    fig, axs = plt.subplots(grid_x, grid_y, sharex  = True, sharey = True, constrained_layout=True, figsize = (12,14))
    
    for i in range(grid_x):
        for j in range(grid_y):
            ant_ind = (i*grid_y)+j
            if ant_ind < len(antennas):
                axs[i,j].plot(frequency_matrix[1,:], phase_dat[ant_ind,2,:], '.',  label = "BD0")
                axs[i,j].plot(frequency_matrix[1,:], phase_dat[ant_ind,3,:], '.',  label = "BD1")
                axs[i,j].set_title(f"{antennas[ant_ind]}_BD")
                axs[i,j].legend(loc = 'upper right')
            
    fig.suptitle(f"Calculated Phase Calibration Coefficients vs Freq from\n{outfilestem}\nfor source {source_name} and tuning BD")
    fig.supylabel("Phase (degrees)")
    fig.supxlabel("Frequency Channels")

    if outfilestem is not None:
        outfile_name = outfilestem+"phaseBD_vs_antennas.png"
    else:
        outfile_name = "phaseBD_vs_antennas.png"

    if outdir is not None:
        phase_file_path_bd = os.path.join(outdir, outfile_name)
    else:
        phase_file_path_bd = outfile_name
         
    plt.savefig(phase_file_path_bd, dpi = 150)
    plt.close() 

    return delay_file_path, phase_file_path_ac, phase_file_path_bd

if __name__ == '__main__':
    
    # Argument parser taking various arguments
    parser = argparse.ArgumentParser(
        description='Reads delay and phase JSON file to make plots',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d','--delay-file', type = str, required = True, help = 'JSON Delay file')
    parser.add_argument('-p','--phase-file', type = str, required = True, help = 'JSON Phase file')
    parser.add_argument('-f','--freq-file', type=str, required = True, help='.npy file containing frequency matrix')
    parser.add_argument('-o','--out-dir', type = str, required = False, default = '.', help = 'Output directory to save the plots')
    args = parser.parse_args()

    #opening each json files
    with open(args.delay_file) as dh:
        delay_dict = json.load(dh)

    with open(args.phase_file) as ph:
        phase_dict = json.load(ph)

    freq_matrix = np.load(args.freq_file)

    plot_delay_phase(delay_dict, phase_dict, freq_matrix, outdir = args.out_dir)
