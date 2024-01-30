from pydantic import BaseModel
import typing
import logging
from logging.handlers import WatchedFileHandler
from enum import Enum
import datetime
import time
from astropy.time import Time
import argparse
import re
import os

LOGFILENAME = "/home/cosmic/logs/CalibrationRetroArchival.log"
logger = logging.getLogger('retro_archiver')
logger.setLevel(logging.DEBUG)

fh = WatchedFileHandler(LOGFILENAME, mode = 'a')
fh.setLevel(logging.DEBUG)

logger.addHandler(fh)

def UnixTimeFromModifiedJulianDate(jd):
    return (jd-40587)*86400000

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

    for root, dirs, files in os.walk(os.path.abspath(args.rootdir)):
        #Check if we are in a directory with uvh5's or a calibration directory
        if "calibration" in dirs or any(file.endswith('.uvh5') for file in os.listdir(root)):
            #We are in scan base directory with calibration folder and uvh5's

            #Check if there are calibration products (phases/delays) derived:
            delays_calculated = os.path.isdir(root + "/calibration/fixed_delays") and len(os.listdir(root + "/calibration/fixed_delays"))==1
            phases_calculated = os.path.isdir(root + "/calibration/fixed_phases") and len(os.listdir(root + "/calibration/fixed_phases"))==1
            if delays_calculated and phases_calculated:
                logger.info(f"Processing contents of directory {root} as there are calibration products in here")
            else:
                logger.error(f"Not processing contents of directory {root} as there are no/partial calibration products in here.\nCould delete this calibration folder")
                continue

            #Check uvh5 files present in root
            if not any(file.endswith('.uvh5') for file in os.listdir(root)):
                logger.warning(f"No UVH5 files present in scan root directory: {root}.\nCalibration products exist but no UVH5 files present.")
                no_uvh5 = True
            else:
                no_uvh5=False
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
                files_to_process = [f for f in os.listdir(root +"/calibration/calibration_gains/") if os.path.isfile(os.path.join(root +"/calibration/calibration_gains/", f))]
            else:
                if no_uvh5:
                    logger.error(f"There are no UVH5 or JSON gains files present in scan root directory: {root}.\nUnable to recreate calibration run...")
                    #LOG TO FILE THAT NO CALIBRATION RUN CAN BE REPLICATED - MISSING UVH5 AND JSON BUT PRODUCTS ARE PRESENT. 
                else:
                    logger.warning(f"There are no JSON files present in scan root directory: {root} but there are UVH5's. Recreating...")
                    uvh5_command = f"/home/cosmic/anaconda3/envs/cosmic_vla/bin/python3 /home/cosmic/dev/COSMIC-VLA-CalibrationEngine/calibrate_uvh5.py -d {root} -o {root + '/calibration'} --gengain"                                
                    logger.debug(f"Executing:\n{uvh5_command}")
                    os.system(uvh5_command)
                    
            #Get some details from one of the json file names:
            observation_details_from_json = None
            directory_contents = os.listdir(root+ "/calibration/calibration_gains")
            for item in directory_contents:
                if os.path.isfile(os.path.join(root+ "/calibration/calibration_gains",item)) and item.endswith('.json'):
                    observation_details_from_json = FilenameParts(item)
                    break

            mjd_time = Time(f"{observation_details_from_json.mjd_day}.{observation_details_from_json.mjd_seconds}",format='mjd')
            t = datetime.datetime.strptime('Mon Apr 10 00:00:00 2023', "%a %b %d %H:%M:%S %Y")
            db_env = ''
            if mjd_time.unix < 1681084800.0:
                db_env = 'COSMIC_DB_TABLE_SUFFIX=_pre_202304010 '                

            #TEMPORARY
            sideband = '1 1'
            tbin = '1e-6'
            fcentmhz = '2477 3501'
            # collate_command = f"{db_env}/home/cosmic/anaconda3/envs/cosmic_vla/bin/python3 /home/cosmic/dev/COSMIC-VLA-DelayEngine/calibration_gain_collator.py {root + '/calibration/calibration_gains'+'/*.json'} -o {root + '/calibration'} --no-slack-post --fcentmhz {fcentmhz} --sideband {sideband} --snr-threshold 4.0 --tbin {tbin}  --start-epoch-seconds {mjd_time.unix} --cosmicdb-engine-configuration /home/cosmic/conf/cosmicdb_conf.yaml --dry-run --archive-mode"
            collate_command = f"/home/cosmic/anaconda3/envs/cosmic_vla/bin/python3 /home/cosmic/dev/COSMIC-VLA-DelayEngine/calibration_gain_collator.py {root + '/calibration/calibration_gains'+'/*.json'} -o {root + '/calibration'} --no-slack-post --fcentmhz {fcentmhz} --sideband {sideband} --snr-threshold 4.0 --tbin {tbin}  --start-epoch-seconds {mjd_time.unix} --dry-run --archive-mode"
            logger.debug(f"Re-collating for retroarchival. Executing:\n{collate_command}")
            os.system(collate_command)
            