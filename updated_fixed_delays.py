import pandas as pd
import argparse
import os

def update_fixed_delays(fixed_csv : str, correction_csv : str):
    """
    Accepts two csv's for fixed delays and its corrections and subtracts the corrections
    from the fixed delays before returning the updated dataframe.
    """
    fixed_delays = pd.read_csv(os.path.abspath(fixed_csv), names = ["IF0","IF1","IF2","IF3"],
                                 header=None, skiprows=1)
    correction_delays = pd.read_csv(os.path.abspath(correction_csv), names = ["IF0","IF1","IF2","IF3"],
                                 header=None, skiprows=1)
    return fixed_delays.subtract(correction_delays)

if __name__=="__main__":
    parser = argparse.ArgumentParser(
    description=("""Taking in a csv file of fixed-delays and a csv file of fixed-delay
                    corrections, produce a new fixed-delays csv for delaycalibration.py to
                    upload.""")
    )
    parser.add_argument("--fixed_csv", type=str, help="fixed delays filepath.")
    parser.add_argument("--correction_csv", type=str, help="fixed delay correction filepath.")
    args = parser.parse_args()

    fixed_delays_df = update_fixed_delays(args.fixed_csv, args.correction_csv)
    fixed_delays_df.to_csv('latest.csv')
