import os
import argparse
import shutil

#add logging to a local file:
import logging
logging.basicConfig(filename='./place_missplaced_uvh5.log', level=logging.INFO)


#helper script that injests a file with all paths not containing uvh5 files (output of retroarchival) and then reads an input directory for uvh5 files and places them accordingly:
PATHS_MISSING_UVH5 = "./no_uvh5.txt"

if __name__ == "__main__":
    #Add argument parser for input directory:
    parser = argparse.ArgumentParser(description='Place missplaced uvh5 files')
    parser.add_argument('-d','--input_dir', type=str, help='Input directory containing uvh5 files')
    args = parser.parse_args()
    input_dir = args.input_dir

    #create a list of all uvh5 files in the input directory:
    uvh5_files = [f for f in os.listdir(input_dir) if f.endswith('.uvh5')]

    #open the file with all paths not containing uvh5 files:
    with open(PATHS_MISSING_UVH5, 'r') as f:
        missplaced_paths = f.readlines()
        path_ids = [path.split('/')[-1] for path in missplaced_paths]
        for uvh5_file in uvh5_files:
            uvh5_id = uvh5_file.split('.')[0:-3]
            uvh5_id = '.'.join(uvh5_id)
            if uvh5_id in path_ids:
                #get the path of the missplaced file:
                path = missplaced_paths[path_ids.index(uvh5_file)].strip()
                destination_file = os.path.join(path, uvh5_file)
                shutil.copy2(os.path.join(input_dir, uvh5_file), destination_file)
                print(f"Copied {uvh5_file} to {path}")
                logging.info(f"Copied {uvh5_file} to {path}")
                # check if the copy was successful
                if os.path.exists(destination_file):
                    # remove the original file
                    os.remove(os.path.join(input_dir, uvh5_file))
                    print(f"Removed {uvh5_file} from {input_dir}")
                    logging.info(f"Removed {uvh5_file} from {input_dir}")
                else:
                    print(f"Failed to copy {uvh5_file}")
                    logging.info(f"Failed to copy {uvh5_file}")
            else:
                print(f"Could not find path for {uvh5_file}")
                logging.info(f"Could not find path for {uvh5_file}")