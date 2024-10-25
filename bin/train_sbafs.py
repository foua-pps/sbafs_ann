#!/usr/bin/env python
import argparse
import glob
from sbafs_ann.sbaf_ann_lib import train_network_for_files


if __name__ == "__main__":
    """ Train NN SBAF"""
    parser = argparse.ArgumentParser(
        description=('Script to train ANN SBAFs'))
    parser.add_argument('channel_list', metavar='channel_listN', type=str, nargs='+',
                        help='List of channels to train with')
    parser.add_argument('--train_dir', type=str, nargs='?',
                        required=True, default='.',
                        help="Directory with NOAA-19 training files.")
    parser.add_argument('--valid_dir', type=str, nargs='?',
                        required=True, default='.',
                        help="Directory with NOAA-19 during training validation files.")
    parser.add_argument('--accept_satz_max', type=int, nargs='?',
                        required=False, default=25,
                        help="Max satz angle")
    parser.add_argument('--accept_sunz_max', type=int, nargs='?',
                        required=False, default=180,
                        help="Max sunz angle")
    parser.add_argument('--accept_sunz_min', type=int, nargs='?',
                        required=False, default=0,
                        help="Min sunz angle")
    parser.add_argument('--accept_time_diff', type=int, nargs='?',
                        required=False, default=5,
                        help="Allowd time difference in minutes")
    parser.add_argument('--max_distance_between_pixels_m', type=int, nargs='?',
                        required=False, default=3000,
                        help="Allowd max distance (m)")
    parser.add_argument('-o', '--output_dir', type=str, nargs='?',
                        required=False, default='.',
                        help="Output directory where to store the train network files")
    options = parser.parse_args()
    files_train = glob.glob(
        "{:s}/matchup_avhrr_*_*viirs*.h5".format(options.train_dir))
    print(files_train)
    files_valid = glob.glob(
        "{:s}/matchup_avhrr_*_*viirs*.h5".format(options.valid_dir))
    train_network_for_files(options, files_train, files_valid)
