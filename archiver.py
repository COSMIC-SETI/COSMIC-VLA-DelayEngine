from pydantic import BaseModel
import typing
import logging
from logging.handlers import WatchedFileHandler
from enum import Enum
import time
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

def checkForCalibrationGains(calibration_dir: str) -> bool:
    """Given a calibration_dir directory, check if the directory contains calibration gains. Return True if it does, else False"""
    subdirs = os.listdir(calibration_dir)
    for subdir in subdirs:
        #check if the subdir is 'calibration_gains':
        if subdir == 'calibration_gains':
            #We have calibration gains, check that it contains .json files:
            if any([re.match(r".*\.json", file) for file in os.listdir(os.path.join(calibration_dir, subdir))]):
                #We have calibration gains:
                logger.info(f"Found calibration gains in '{root}'")
                return True
            else:
                #We have a calibration gains folder but no gains:
                logger.info(f"Found calibration_gains folder in '{root}', but no calibration gains therein.")
                return False
        else:
            continue

    return False

def writeCalibrationCollationCommand(uvh5_ac: str, uvh5_bd: str) -> (str, bool):
    """Given a uvh5 file of each tuning in the same root, derive a collation command and return it.
    Return command and Ture if the command could be derived. Else return error message and False."""
    #read the uvh5 files:
    try:
        uv.read(uvh5_ac)
        start_epoch_seconds = Time(f"{uv.time_array[0] - 2400000.5}",format='mjd') 
        fcentmhz_ac = uv.extra_keywords['CenterFrequencyHz']*1e-6
        tbin=1e-6
        sideband_ac = -1 if (fcentmhz_ac < 12000 and fcentmhz_ac > 8000) else 1
    except Exception as e:
        logger.error(f"Could not read uvh5 file '{uvh5_ac}' or extract required information: {e}")
        return f"Could not read uvh5 file '{uvh5_ac}' or extract required information: {e}", False

    try:
        uv.read(uvh5_bd)
        fcentmhz_bd = uv.extra_keywords['CenterFrequencyHz']*1e-6
        sideband_bd = -1 if (fcentmhz_bd < 12000 and fcentmhz_bd > 8000) else 1
    except Exception as e:
        logger.error(f"Could not read uvh5 file '{uvh5_bd}' or extract required information: {e}")
        return f"Could not read uvh5 file '{uvh5_bd}' or extract required information: {e}", False
    
    #derive database environment variable:
    if start_epoch_seconds.unix > 1685810556.0: #we've already recorded gains from this point onwards, so ignore.
        logger.info(f"Scan should have already been archived. Skipping...")
        return f"Scan is newer than {time.ctime(1685810556.0)} and should have already been archived. Skipping...", False
    if start_epoch_seconds.unix < 1681084800.0:
        env['COSMIC_DB_TABLE_SUFFIX']='_pre_20230410'
    else:
        if "COSMIC_DB_TABLE_SUFFIX" in env:
            del env['COSMIC_DB_TABLE_SUFFIX']

    if os.path.dirname(uvh5_ac) == os.path.dirname(uvh5_bd):
        root = os.path.dirname(uvh5_ac)
    else:
        logger.error(f"Both tuning UVH5 files are not in the same directory.")
        return f"UVH5 files are not in the same directory.", False

    #derive the command:
    collate_command = f"/home/cosmic/anaconda3/envs/cosmic_vla/bin/python3 /home/cosmic/dev/COSMIC-VLA-DelayEngine/calibration_gain_collator.py {root + '/calibration/calibration_gains'+'/*.json'} -o {root + '/calibration'} --no-slack-post --fcentmhz {fcentmhz_ac} {fcentmhz_bd} --sideband {sideband_ac} {sideband_bd} --snr-threshold 4.0 --tbin {tbin}  --start-epoch-seconds {start_epoch_seconds.unix} --cosmicdb-engine-configuration /home/cosmic/conf/cosmicdb_conf.yaml --dry-run --archive-mode"
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
                        collate_command, status = writeCalibrationCollationCommand(ac_file, bd_file)

                        if status == False:
                            logger.warning(f"Error deriving collation command: {collate_command}")
                            csvwriter.writerow([timestamp, observation_name, False, collate_command, root])
                            continue
                            
                        if cal_products:
                            #We have calibration products and uvh5 files, check for calibration gains:
                            have_gains = checkForCalibrationGains(calibration_dir)
                            if not have_gains:
                                #We have calibration products but no gains, re-derive gains:
                                logger.debug(f"Found calibration products in '{root}', but no calibration gains, re-deriving gains.")
                                uvh5_command = f"/home/cosmic/anaconda3/envs/cosmic_vla/bin/python3 /home/cosmic/dev/COSMIC-VLA-CalibrationEngine/calibrate_uvh5.py {root} --gengain"
                                try:
                                    logger.debug(f"Executing:\n{uvh5_command}")
                                    uvh5_process = subprocess.Popen(uvh5_command, shell=True)
                                    uvh5_process.wait() 
                                except subprocess.CalledProcessError:
                                    logger.error(f"Error executing:\n{uvh5_command}")
                                    csvwriter.writerow([timestamp, observation_name, False, "Error recreating gains, unable to proceed", root])
                                    continue

                            #Now recheck gains exist:
                            have_gains = checkForCalibrationGains(calibration_dir)
                            if not have_gains:
                                logger.error(f"Re-derivation of calibration gains failed.")
                                csvwriter.writerow([timestamp, observation_name, False, "Re-derivation of calibration gains failed. It is possibly a permissions or disk space error in writing out gains.", root])
                                continue
                            
                            #By this point we have calibration products and calibration gains, proceed with collation:
                            try:
                                logger.debug(f"Re-collating for retroarchival. Executing:\n{collate_command}")
                                collate_process = subprocess.Popen(collate_command, shell=True, env=env)
                                collate_process.wait()
                            except subprocess.CalledProcessError:
                                logger.error(f"Error executing:\n{collate_command}")
                                csvwriter.writerow([timestamp, observation_name, False, "Could not collate gains and calculate grade.", root])
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
