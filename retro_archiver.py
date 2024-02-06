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
import csv

LOGFILENAME = "/home/cosmic/logs/CalibrationRetroArchival.log"
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        """Pass in root directory. Cannot pass in a glob directory structure (i.e. no *)"""
    )
    parser.add_argument('-r', '--rootdir', type=str, help="String root path entry")
    args = parser.parse_args()
    uv = UVData() #create uv instance for reading uvh5 files.
    #open a csv file (append if it exists) for logging, create the file if it does not and write the header

    #create CSVLOG file if it does not exist and write header to it:
    if not os.path.isfile(CSVLOG):
        with open(CSVLOG, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            #write header 'timestamp', 'scan_id', 'archival_status', 'notes', 'data_path':
            csvwriter.writerow(['timestamp', 'scan_id', 'archival_status', 'notes', 'data_path'])

    with open(CSVLOG, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)

        for root, dirs, files in os.walk(os.path.abspath(args.rootdir)):
            #Check if we are in a directory with uvh5's or a calibration directory
            if "calibration" in dirs or any(file.endswith('.uvh5') for file in os.listdir(root)):
                #We are in scan base directory with calibration folder and uvh5's
                timestamp = time.time() #generate timestamp for csv
                archival_status = False
                notes = ''
                scan_id = root.split('/')[-1]
                #Check if there are calibration products (phases/delays) derived:
                delays_calculated = os.path.isdir(root + "/calibration/fixed_delays") and len(os.listdir(root + "/calibration/fixed_delays"))>0
                phases_calculated = os.path.isdir(root + "/calibration/fixed_phases") and len(os.listdir(root + "/calibration/fixed_phases"))>0
                if delays_calculated and phases_calculated:
                    logger.info(f"Processing contents of directory {root} as there are calibration products in here")
                else:
                    logger.error(f"Not processing contents of directory {root} as there are no/partial calibration products in here.\nCould delete this calibration folder")
                    notes = f"Did not calibrate"
                    csvwriter.writerow([timestamp, scan_id, archival_status, notes, root])
                    continue

                #Check uvh5 files present in root
                if not any(file.endswith('.uvh5') for file in os.listdir(root)):
                    logger.warning(f"No UVH5 files present in scan root directory: {root}.\nCalibration products exist but no UVH5 files present.")
                    no_uvh5 = True
                    mjd_time=None
                    notes = "No UVH5 files found"
                else:
                    no_uvh5=False
                    #fetch time details from the first uvh5 file:
                    uvh5_file = [file for file in os.listdir(root) if file.endswith('.uvh5')][0]
                    try:
                        uv.read(os.path.join(root,uvh5_file))
                        mjd_time = uv.time_array[0]
                    except:
                        logger.error(f"Error reading uvh5 file: {os.path.join(root,uvh5_file)}")
                        continue
                    logger.info(f"There are UVH5 files present in scan root directory: {root}")
                    number_of_uvh5_files = len([f for f in os.listdir(root) if (os.path.isfile(os.path.join(root, f)) and f.endswith('.uvh5'))])
                    logger.debug(f"Number of UVH's present: {number_of_uvh5_files}")

                #Check for calibration gains in root
                if "calibration" in dirs and os.path.isdir(root + "/calibration/calibration_gains"):
                    logger.info(f"There are JSON files present in scan root directory: {root}")
                    number_of_json_files = len([f for f in os.listdir(root+ "/calibration/calibration_gains")])
                    logger.debug(f"Number of JSON's present: {number_of_json_files}")
                    #We derived calibration solutions, collate.
                    logger.info(f"{100*number_of_json_files/number_of_uvh5_files}% of the gains were generated from uvh5's.")
                else:
                    if no_uvh5:
                        logger.error(f"There are no UVH5 or JSON gains files present in scan root directory: {root}.\nUnable to recreate calibration run...")
                        notes = "No UVH5 or JSON files found"
                        csvwriter.writerow([timestamp, scan_id, archival_status, notes, root])
                        continue
                        #LOG TO FILE THAT NO CALIBRATION RUN CAN BE REPLICATED - MISSING UVH5 AND JSON BUT PRODUCTS ARE PRESENT. 
                    else:
                        logger.warning(f"There are no JSON files present in scan root directory: {root} but there are UVH5's. Recreating...")
                        uvh5_command = f"/home/cosmic/anaconda3/envs/cosmic_vla/bin/python3 /home/cosmic/dev/COSMIC-VLA-CalibrationEngine/calibrate_uvh5.py {root} --gengain"                                
                        try:
                            logger.debug(f"Executing:\n{uvh5_command}")
                            os.system(uvh5_command)
                        except:
                            logger.error(f"Error executing:\n{uvh5_command}")
                            notes = "Error recreating gains"
                            csvwriter.writerow([timestamp, scan_id, archival_status, notes, root])
                            continue
                        
                if not mjd_time:
                    #Get some details from one of the json file names to use as database time since no uvh5 files are present to extract it from.
                    observation_details_from_json = None
                    directory_contents = os.listdir(root+ "/calibration/calibration_gains")
                    for item in directory_contents:
                        if os.path.isfile(os.path.join(root+ "/calibration/calibration_gains",item)) and item.endswith('.json'):
                            observation_details_from_json = FilenameParts(item)
                            break
                    start_epoch_seconds = Time(f"{observation_details_from_json.mjd_day}.{observation_details_from_json.mjd_seconds}",format='mjd')
                else:
                    start_epoch_seconds = Time(f"{mjd_time-2400000.5}",format='mjd')

                if start_epoch_seconds.unix > 1685810556.0: #we've already recorded gains from this point onwards, so ignore.
                    logger.info(f"Scan {scan_id} should have already been archived. Skipping...")
                    continue
                else:                
                    db_env = ''
                    if start_epoch_seconds.unix < 1681084800.0:
                        db_env = 'COSMIC_DB_TABLE_SUFFIX=_pre_20230410 '
                    #TEMPORARY
                    sideband = '1 1'
                    tbin = '1e-6'
                    fcentmhz = '2477 3501'
                    collate_command = f"{db_env}/home/cosmic/anaconda3/envs/cosmic_vla/bin/python3 /home/cosmic/dev/COSMIC-VLA-DelayEngine/calibration_gain_collator.py {root + '/calibration/calibration_gains'+'/*.json'} -o {root + '/calibration'} --no-slack-post --fcentmhz {fcentmhz} --sideband {sideband} --snr-threshold 4.0 --tbin {tbin}  --start-epoch-seconds {start_epoch_seconds.unix} --cosmicdb-engine-configuration /home/cosmic/conf/cosmicdb_conf.yaml --dry-run --archive-mode"
                    # collate_command = f"/home/cosmic/anaconda3/envs/cosmic_vla/bin/python3 /home/cosmic/dev/COSMIC-VLA-DelayEngine/calibration_gain_collator.py {root + '/calibration/calibration_gains'+'/*.json'} -o {root + '/calibration'} --no-slack-post --fcentmhz {fcentmhz} --sideband {sideband} --snr-threshold 4.0 --tbin {tbin}  --start-epoch-seconds {start_epoch_seconds} --dry-run --archive-mode"
                    try:
                        logger.debug(f"Re-collating for retroarchival. Executing:\n{collate_command}")
                        os.system(collate_command)
                    except:
                        logger.error(f"Error executing:\n{uvh5_command}")
                        notes = "Error collating gains"
                        csvwriter.writerow([timestamp, scan_id, archival_status, notes, root])
                        continue
                    archival_status = True
                    csvwriter.writerow([timestamp, scan_id, archival_status, notes, root])
                