from pydantic import BaseModel
import typing
import logging
from logging.handlers import WatchedFileHandler
from enum import Enum
import time
import json
import numpy as np
from astropy.time import Time
from pyuvdata import UVData
import argparse
import re
import os
import subprocess
import csv

env = os.environ.copy()

LOGFILENAME = "/home/cosmic/logs/CalibrationArchival.log"
# LOGFILENAME = "./retroarchival.log"
CSVLOG = '/home/cosmic/logs/retroarchival.csv'
# CSVLOG = './retroarchival.csv'
logger = logging.getLogger('retro_archiver')
logger.setLevel(logging.DEBUG)

fh = WatchedFileHandler(LOGFILENAME, mode = 'a')
fh.setLevel(logging.DEBUG)

logger.addHandler(fh)

class LoTuningName(str, Enum):
    AC = "AC"
    BD = "BD"

class Extension(str, Enum):
    raw = "raw"
    bfr5 = "bfr5"
    stamps = "stamps"
    hits = "hits"
    uvh5 = "uvh5"
    json = "json"

class FilenameParts(BaseModel):
    filename: str
    scan_project_name: str
    mjd_day: str
    mjd_seconds: str
    scan_no: int
    subscan_no: int
    tuning: LoTuningName
    schan: int
    rawpart_enumeration: typing.Optional[int]
    suffix: typing.Optional[str]
    extension: Extension

    observation_id: str

    def __init__(self, filename: str):
        fname, ext = os.path.splitext(filename)
        m = re.match(
            r"(?P<scan_proj>.*)\.(?P<mjdd>\d+)\.(?P<mjds>\d+)\.(?P<scan_no>\d+)\.(?P<subscan_no>\d+)\.(?P<tuning>\w\w)\.C(?P<schan>\d+)(\.(?P<rawpart>\d+)(\.(?P<suffix>.*))?)?",
            fname
        )
        if m is None:
            raise ValueError(f"Could not recognise filename form '{filename}'")
        
        super().__init__(
            filename=filename,
            scan_project_name=m.group("scan_proj"),
            mjd_day=m.group("mjdd"),
            mjd_seconds=m.group("mjds"),
            scan_no=int(m.group("scan_no")),
            subscan_no=int(m.group("subscan_no")),
            tuning=m.group("tuning"),
            schan=int(m.group("schan")),
            rawpart_enumeration=int(m.group("rawpart")) if m.group("rawpart") is not None else None,
            suffix=m.group("suffix"),
            extension=ext[1:],
            observation_id=".".join([
                m.group("scan_proj"),
                m.group("mjdd"),
                m.group("mjds"),
                m.group("scan_no"),
                m.group("subscan_no")
            ])
        )

def setupCSVLOG() -> None:
    """Create CSVLOG file if it does not exist and write header to it:"""
    if not os.path.isfile(CSVLOG):
        with open(CSVLOG, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            #write header 'timestamp', 'scan_id', 'archival_status', 'notes', 'data_path':
            csvwriter.writerow(['timestamp', 'obs_id', 'archival_status', 'notes', 'data_path'])
    else:
        pass

def checkUVH5Uniformity(files : list) -> str:
    """Given a list of files, iterate through them and assert that all files
    are of the same observation. If they are not, return False, else return True."""
    #check files are a list of strings:
    if not all(isinstance(file, str) for file in files):
        raise ValueError("Files must be a list of strings")
    observation_name = None
    for file in files:
        #read the fileparts:
        try:
            fparts = FilenameParts(file)
        except ValueError as e:
            logger.error(f"Could not parse filename '{file}': {e}")
            continue
        #check if its is uvh5:
        if fparts.extension == 'uvh5':
            #We have uvh5 files, this is an assumed calibration directory:
            if observation_name is None:
                observation_name = fparts.observation_id
            elif fparts.observation_id != observation_name:
                #We have mixed observation uvh5 files in here - log and continue:
                logger.warning(f"Found mixed observation uvh5 files in directory.")
                return ''
        
    if observation_name is None:
        #This means all files could not be parsed
        logger.error(f"Unable to parse or find uvh5 files in directory.")
        return ''
    else:
        return observation_name

def checkForCalibrationProducts(calibration_dir: str) -> bool:
    """Given a calibration_dir directory, check if the directory contains calibration products. Return True if it does, else False"""
    subdirs = os.listdir(calibration_dir)
    if all(subdir in subdirs for subdir in ['fixed_delays','fixed_phases']):
        # All critical subdirectories are present
        logger.info(f"Found calibration directory in '{root}', with all critical subdirectories. Calibration did occur.")
        return True
    else:
        logger.info(f"Found calibration directory in '{root}', but missing critical subdirectories. Skipping.")
        return False

def checkCalibrationGains(calibration_dir: str) -> bool:
    """Given a calibration_dir directory, check if the directory contains calibration gains and check that they contain a reference antenna. 
    Return True if it does, else False"""
    subdirs = os.listdir(calibration_dir)
    for subdir in subdirs:
        #check if the subdir is 'calibration_gains':
        if subdir == 'calibration_gains':
            #We have calibration gains, check that it contains .json files:
            if any([re.match(r".*\.json", file) for file in os.listdir(os.path.join(calibration_dir, subdir))]):
                #We have calibration gains:
                logger.info(f"Found calibration gains in '{root}'")
                fname = f"{calibration_dir}/calibration_gains/{os.listdir(calibration_dir+'/calibration_gains/')[0]}"
                with open(fname, 'r') as f:
                    try:
                        gains = json.load(f)
                    except:
                        logger.error(f"Error reading calibration gains from file {fname}.")
                        return False
                    if 'ref_ant' not in gains[list(gains.keys())[0]]:
                        logger.info(f"Found calibration gains in '{root}', but no reference antenna field.")
                        return False
                    else:
                        logger.info(f"Found calibration gains in '{root}', with reference antenna field.")
                        return True
            else:
                #We have a calibration gains folder but no gains:
                logger.info(f"Found calibration_gains folder in '{root}', but no calibration gains therein.")
                return False
        else:
            continue

    return False

def determineReferenceAntenna(calibration_dir: str, obs_info: dict) -> str:
    """Given the calibration directory and observation information, look at the antenna will
    closest to zero phases that is within the observation antennas. Return the antenna name. Return
    None if failure."""
    #Fetch the fixed_phases json filename:
    json_file = os.path.join(calibration_dir,f"fixed_phases/{os.listdir(os.path.join(calibration_dir,'fixed_phases/'))[0]}")

    # Load the JSON file
    with open(json_file, 'r') as f:
        fixed_phases = json.load(f)

    min_sum = None
    min_antenna = None

    # Iterate over the dictionary
    for antenna_name, lists in fixed_phases.items():
        if antenna_name in obs_info["observed_antenna"]:
            # Flatten the list of lists and calculate the absolute sum
            total = np.sum(np.abs(np.array(lists)))
            
            # Update min_sum and min_antenna if necessary
            if min_sum is None or total < min_sum:
                min_sum = total
                min_antenna = antenna_name

    return min_antenna

def fetchUVH5detail(uvh5_ac: str, uvh5_bd: str) -> (dict, bool):
    """Given a uvh5 file of each tuning in the same root, fetch details from the observation and return a dict.
    This dictionary contains all necessary information for the uvh5 calibration and collation commands."""
    obs_info = {}
    if os.path.dirname(uvh5_ac) == os.path.dirname(uvh5_bd):
        obs_info["root"] = os.path.dirname(uvh5_ac)
    else:
        logger.error(f"Both tuning UVH5 files are not in the same directory.")
        return f"UVH5 files are not in the same directory.", False

    try:
        uv.read(uvh5_ac, fix_old_proj=False)
        obs_info["start_epoch_seconds"] = Time(f"{uv.time_array[0] - 2400000.5}",format='mjd').unix 
        obs_info["sideband_ac"] = -1 if (uv.extra_keywords['FirstChannelFrequencyHz'] < 12000 and uv.extra_keywords['FirstChannelFrequencyHz'] > 8000) else 1
        if obs_info["sideband_ac"] == -1:
            obs_info["fcentmhz_ac"] = (uv.extra_keywords['FirstChannelFrequencyHz'] - (uv.extra_keywords['NumberOfFEngineChannels']//2)*uv.extra_keywords['ChannelBandwidthHz'])*1e-6
        else:
            obs_info["fcentmhz_ac"] = (uv.extra_keywords['FirstChannelFrequencyHz'] + (uv.extra_keywords['NumberOfFEngineChannels']//2)*uv.extra_keywords['ChannelBandwidthHz'])*1e-6
        obs_info["tbin"] = 1e-6

        #fetch all antenna names present in the observation:
        ants_numbers = uv.get_ants()
        obs_info["observed_antenna"] = [uv.antenna_names[uv.antenna_numbers.tolist().index(antnum)] for antnum in ants_numbers]

    except Exception as e:
        logger.error(f"Could not read uvh5 file '{uvh5_ac}' or extract required information: {e}")
        return f"Could not read uvh5 file '{uvh5_ac}' or extract required information: {e}", False

    try:
        uv.read(uvh5_bd, fix_old_proj=False)
        obs_info["sideband_bd"] = -1 if (uv.extra_keywords['FirstChannelFrequencyHz'] < 12000 and uv.extra_keywords['FirstChannelFrequencyHz'] > 8000) else 1
        if obs_info["sideband_bd"] == -1:
            obs_info["fcentmhz_bd"] = (uv.extra_keywords['FirstChannelFrequencyHz'] - (uv.extra_keywords['NumberOfFEngineChannels']//2)*uv.extra_keywords['ChannelBandwidthHz'])*1e-6
        else:
            obs_info["fcentmhz_bd"] = (uv.extra_keywords['FirstChannelFrequencyHz'] + (uv.extra_keywords['NumberOfFEngineChannels']//2)*uv.extra_keywords['ChannelBandwidthHz'])*1e-6
    except Exception as e:
        logger.error(f"Could not read uvh5 file '{uvh5_bd}' or extract required information: {e}")
        return f"Could not read uvh5 file '{uvh5_bd}' or extract required information: {e}", False

    return obs_info, True

def writeUVH5CalibrationCommand(obs_info : dict) -> (str, bool):
    """Given observation information derive a calibration command and return it alongside a status.
    """
    #derive the command:
    if obs_info["reference_antenna"] is None:
        uvh5_command = f"/home/cosmic/anaconda3/envs/cosmic_vla/bin/python3 /home/cosmic/dev/COSMIC-VLA-CalibrationEngine/calibrate_uvh5.py {obs_info['root']} --gengain"
    else:
        uvh5_command = f"/home/cosmic/anaconda3/envs/cosmic_vla/bin/python3 /home/cosmic/dev/COSMIC-VLA-CalibrationEngine/calibrate_uvh5.py {obs_info['root']} --gengain --refant={obs_info['reference_antenna']}"
    return uvh5_command, True

def writeCalibrationCollationCommand(obs_info : dict, fcent_offset : float=0.0) -> (str, bool):
    """Given observation information, derive a collation command and return it.
    Return command and Ture if the command could be derived. Else return error message and False.
    
    There is an option to provide an fcent offset due to occasionally incorrectly written frequency information in the UVH5 files"""
    
    #derive database environment variable:
    if obs_info["start_epoch_seconds"] > 1685810556.0: #we've already recorded gains from this point onwards, so ignore.
        logger.info(f"Scan should have already been archived. Skipping...")
        return f"Scan is newer than {time.ctime(1685810556.0)} and should have already been archived. Skipping...", False
    if obs_info["start_epoch_seconds"] < 1681084800.0:
        env['COSMIC_DB_TABLE_SUFFIX']='_pre_20230410'
    else:
        if "COSMIC_DB_TABLE_SUFFIX" in env:
            del env['COSMIC_DB_TABLE_SUFFIX']

    #derive the command:
    collate_command = f"/home/cosmic/anaconda3/envs/cosmic_vla/bin/python3 /home/cosmic/dev/COSMIC-VLA-DelayEngine/calibration_gain_collator.py {obs_info['root'] + '/calibration/calibration_gains'+'/*.json'} -o {obs_info['root'] + '/calibration'} --no-slack-post --fcentmhz {obs_info['fcentmhz_ac']+(fcent_offset)} {obs_info['fcentmhz_bd']+(fcent_offset)} --sideband {obs_info['sideband_ac']} {obs_info['sideband_bd']} --snr-threshold 4.0 --tbin {obs_info['tbin']}  --start-epoch-seconds {obs_info['start_epoch_seconds']} --cosmicdb-engine-configuration /home/cosmic/conf/cosmicdb_conf.yaml --dry-run --archive-mode"
    return collate_command, True

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        """Pass in root directory. Cannot pass in a glob directory structure (i.e. no *)"""
    )
    parser.add_argument('-r', '--rootdir', type=str, help="Root directory to search for files", required=True)
    args = parser.parse_args()
    uv = UVData() #create uv instance for reading uvh5 files.
    timestamp = time.asctime()

    setupCSVLOG()

    with open(CSVLOG, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        #start path traversal loop:
        for root, dirs, files in os.walk(os.path.abspath(args.rootdir)):

            #Check if root contains a "calibration" directory:
            if 'calibration' in dirs:
                calibration_dir = os.path.join(root, 'calibration')
                cal_products = checkForCalibrationProducts(calibration_dir)
                if cal_products:
                    logger.info(f"Found calibration products in {root}")
                else:
                    csvwriter.writerow([timestamp, "did not calibrate", False, "Directory contains no calibration products - therefore unsure calibration was ever run.", root])
                    continue

                if any([re.match(r".*\.uvh5", file) for file in files]):
                    #Root contains uvh5 files, check if they are uniform:
                    observation_name = checkUVH5Uniformity(files)
                    if len(observation_name) == 0:
                        logger.error(f"UVH5 files in directory '{root}' are not uniform or parseable.")
                        csvwriter.writerow([timestamp, "corrupt_uvh5", False, "UVH5 files in directory are not uniform.", root])
                        continue
                    else:
                        #All UVH5 are uniform and the of an appropriate format. Continue with archival:
                        logger.info(f"UVH5 files in directory '{root}' are uniform and parseable.")
                        #read two uvh5 files, one for each tuning AC and BD:
                        ac_file, bd_file = None, None
                        for file in files:
                            if ac_file is not None and bd_file is not None:
                                break
                            if re.match(r".*\.AC.*\.uvh5", file):
                                ac_file = os.path.join(root, file)
                            elif re.match(r".*\.BD.*\.uvh5", file):
                                bd_file = os.path.join(root, file)
                            else:
                                continue

                        if ac_file is None or bd_file is None:
                            logger.error(f"Could not find uvh5 files for both tunings in '{root}'.")
                            csvwriter.writerow([timestamp, observation_name, False, "Could not find at least 1 UVH5 file for both tunings.", root])
                            continue

                        obs_info, status = fetchUVH5detail(ac_file, bd_file)
                        if status == False:
                            logger.warning(f"Error extracting observation information from: {ac_file} and {bd_file}")
                            csvwriter.writerow([timestamp, observation_name, False, obs_info, root])
                            continue

                        obs_info["reference_antenna"] = determineReferenceAntenna(calibration_dir, obs_info)

                        collate_command, status = writeCalibrationCollationCommand(obs_info)
                        if status == False:
                            logger.warning(f"Error generating collation command: {collate_command}")
                            csvwriter.writerow([timestamp, observation_name, False, collate_command, root])
                            continue
                            
                        if cal_products:
                            #We have calibration products and uvh5 files, check for calibration gains:
                            have_gains = checkCalibrationGains(calibration_dir)
                            if not have_gains:
                                #We have calibration products but no gains or incomplete gains, re-derive gains:
                                logger.debug(f"Found calibration products in '{root}', but no calibration gains, re-deriving gains.")
                                uvh5_command, _ = writeUVH5CalibrationCommand(obs_info)
                            
                                logger.debug(f"Executing:\n{uvh5_command}")
                                uvh5_process = subprocess.Popen(uvh5_command, shell=True)
                                exit_code = uvh5_process.wait() 
                                if exit_code != 0:
                                    logger.error(f"Error executing:\n{uvh5_command}")
                                    csvwriter.writerow([timestamp, observation_name, False, "Error recreating gains, unable to proceed", root])
                                    continue
                                else:
                                    #Now recheck gains exist- even if the command was successful, the gains may not have been written out:
                                    have_gains = checkCalibrationGains(calibration_dir)
                                    if not have_gains:
                                        logger.error(f"Re-derivation of calibration gains failed.")
                                        csvwriter.writerow([timestamp, observation_name, False, "Re-derivation of calibration gains failed. It is possibly a permissions or disk space error in writing out gains", root])
                                        continue
                            else:
                                logger.info(f"Found calibration gains in '{root}', with reference antenna field. Not re-deriving.")
                            
                            #By this point we have calibration products and calibration gains, proceed with collation:                       
                            logger.debug(f"Re-collating for retroarchival. Executing:\n{collate_command}")
                            collate_process = subprocess.Popen(collate_command, shell=True, env=env)
                            exit_code = collate_process.wait()
                            if exit_code != 0:
                                logger.debug(f"Error executing:\n{collate_command}, trying different fcent offsets.")
                                fcent_offset_to_try = [0.5,-0.5]
                                for fcent_offset in fcent_offset_to_try:
                                    logger.debug(f"Trying offset {fcent_offset} MHz.")
                                    collate_command, status = writeCalibrationCollationCommand(obs_info, fcent_offset=fcent_offset)
                                    if status == False:
                                        logger.warning(f"Error generating collation command: {collate_command}")
                                        csvwriter.writerow([timestamp, observation_name, False, collate_command, root])
                                        continue
                                    logger.debug(f"Re-collating for retroarchival. Executing:\n{collate_command}")
                                    collate_process = subprocess.Popen(collate_command, shell=True, env=env)
                                    exit_code = collate_process.wait()
                                    if exit_code ==0:
                                        break

                            if exit_code != 0:
                                logger.error(f"Error executing:\n{collate_command}")
                                csvwriter.writerow([timestamp, observation_name, False, f"Could not collate gains and calculate grade. This is likely due to observation not being present in Observation database", root])
                                continue

                            #We have successfully collated gains:
                            logger.info(f"Calibration grade archival successful")
                            csvwriter.writerow([timestamp, observation_name, True, "Calibration grade archival successful", root]) 

                        else:
                            logger.warning(f"Found UVH5 files in '{root}', but no calibration products.")
                            csvwriter.writerow([timestamp, observation_name, False, "Found UVH5 files in '{root}', but no calibration products.", root])
                
                else:
                    logger.warning(f"No UVH5 files found in '{root}'.")
                    csvwriter.writerow([timestamp, "no_uvh5", False, "No UVH5 files found in directory.", root])
                    continue   

            #see if it is not to be processed or contains uvh5's
            else:
                if any([re.match(r".*\.uvh5", file) for file in files]):
                    logger.warning(f"No calibration directory found in '{root}', but does contain uvh5 files. Skipping.")
                    csvwriter.writerow([timestamp, "no calibration", False, "No calibration directory found but does contain uvh5's.", root])
                    continue
                else:
                    logger.info(f"Skipping '{root}' as it does not contain calibration products or uvh5 files.")
                    continue   
