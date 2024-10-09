#!/usr/bin/env python
import argparse
import glob
from sbafs_ann.sbaf_ann_lib import apply_network_and_plot

if __name__ == "__main__":
    """ Apply network and make some plots."""
    parser = argparse.ArgumentParser(
        description=('Apply network and plot'))

    parser.add_argument('channel_list', metavar='fileN', type=str, nargs='+',
                        help='List of channels to train with')
    parser.add_argument('--plot_dir', type=str, nargs='?',
                        required=False, default='.',
                        help="Output directory where to store the level1c file")
    parser.add_argument('--nn_dir', type=str, nargs='?',
                        required=False, default='.',
                        help="NN directory where nn files are stored")   
    parser.add_argument('--n19_test', type=str, nargs='?',
                        required=True, default='.',
                        help="Directory with N19 test files.")
    parser.add_argument('--viirs_dir', type=str, nargs='?',
                        required=True, default='.',
                        help="Directory with VIIS files.")
    parser.add_argument('--vgac_dir', type=str, nargs='?',
                        required=True, default='.',
                        help="Directory with SBAF correced level1c files.")
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
    options = parser.parse_args()
    n19_files = glob.glob("{:s}/S_NWC_avhrr_noaa19_*T*.nc".format(options.n19_test))
    viirs_files = glob.glob("{:s}/S_NWC_viirs_npp_*T*.nc".format(options.viirs_dir))
    vgac_files =  glob.glob("{:s}/S_NWC_avhrr_vgacsnpp_*T*.nc".format(options.vgac_dir))
    apply_network_and_plot(options, n19_files, viirs_files, vgac_files)



          
