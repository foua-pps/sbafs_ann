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
    parser.add_argument('--n19_train', type=str, nargs='?',
                        required=True, default='.',
                        help="Directory with NOAA-19 training files.")
    parser.add_argument('--n19_valid', type=str, nargs='?',
                        required=True, default='.',
                        help="Directory with NOAA-19 during training validation files.")
    parser.add_argument('--viirs_dir', type=str, nargs='?',
                        required=True, default='.',
                        help="Directory with all VIIRS files.")
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
    n19_files_train = glob.glob("{:s}/S_NWC_avhrr_noaa19_*T*.nc".format(options.n19_train))
    n19_files_valid = glob.glob("{:s}/S_NWC_avhrr_noaa19_*T*.nc".format(options.n19_valid))                         
    viirs_files = glob.glob("{:s}/S_NWC_viirs_npp_*T*.nc".format(options.viirs_dir))
    train_network_for_files(options, n19_files_train, n19_files_valid, viirs_files)

          
