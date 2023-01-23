"""
Written by Savin Shynu Varghese and 
edited by Talon Myburgh.
22/01/2023
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse


def plot_delay_phase(residual_delays_dict, residual_phase_dict, frequency_matrix, outdir=None, outfilestem=None):
    """
    Function to collect the delay and phase information
    from the residual files generated after the calibration step
    and plot them.

    Args:
        residual_delays_dict : a dictionary mapping of {ant: [nof_streams, 1]}
        residual_phase_dict  : a dictionary mapping of {ant: [nof_streams, nof_frequencies]} 
        frequency_matrix     : full observation channel frequencies
        outdir               : output directory to which to save the plots
        outfilestem          : string to prefix to the plot filename
    """

    #Getting the antenna info from the keys
    antennas = list(residual_delays_dict.keys())

    if antennas is not None:
        test_delay = residual_delays_dict[antennas[0]]
        test_phase = residual_phase_dict[antennas[0]]
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
        phase_dat[i,...] = residual_phase_dict[ant]
    
    #converting the delay to ns
    delay_dat *= 1e+9
    
    #plotting the residual delays vs antennas
    fig, ax = plt.subplots(constrained_layout=True, figsize = (10,6))

    ax.plot(delay_dat[:,0], '.',  label = "AC0")
    ax.plot(delay_dat[:,1], '.',  label = "AC1")
    ax.plot(delay_dat[:,2], '.',  label = "BD0")
    ax.plot(delay_dat[:,3], '.',  label = "BD1")

    ax.set_title("Residual Delays vs Antennas")
    ax.legend(loc = 'upper right')
    ax.set_xticks(np.arange(len(antennas)))
    ax.set_xticklabels(antennas)
    
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
                axs[i,j].plot(frequency_matrix[1,:], phase_dat[ant_ind,2,:], '.',  label = "BD0")
                axs[i,j].plot(frequency_matrix[1,:], phase_dat[ant_ind,3,:], '.',  label = "BD1")
                axs[i,j].set_title(f"{antennas[ant_ind]}")
                axs[i,j].legend(loc = 'upper right')
            
    fig.suptitle("Residual Phase vs Freq ")
    fig.supylabel("Residual Phase (degrees)")
    fig.supxlabel("Frequency Channels")

    if outfilestem is not None:
        outfile_name = outfilestem+"phase_vs_antennas.png"
    else:
        outfile_name = "phase_vs_antennas.png"

    if outdir is not None:
        phase_file_path = os.path.join(outdir, outfile_name)
    else:
        phase_file_path = outfile_name
         
    plt.savefig(phase_file_path, dpi = 150)
    plt.close() 

    return delay_file_path, phase_file_path

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
