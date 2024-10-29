#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024- sbaf_ann developers
#
# This file is part of sbaf_ann.
#
# atrain_match is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# atrain_match is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with atrain_match.  If not, see <http://www.gnu.org/licenses/>.

import argparse
import glob
from sbafs_ann.sbaf_ann_lib import apply_network_and_plot

if __name__ == "__main__":
    """ Apply network and make some plots."""
    parser = argparse.ArgumentParser(
        description=('Apply network and plot'))
    parser.add_argument('--plot_dir', type=str, nargs='?',
                        required=False, default='.',
                        help="Output directory where to store the level1c file")
    parser.add_argument('--nn_cfg_file', type=str, nargs="?",
                        required=True,
                        help="NN config file")
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
                        required=False, default=None,
                        help="Max satz angle")
    parser.add_argument('--accept_sunz_max', type=int, nargs='?',
                        required=False, default=None,
                        help="Max sunz angle")
    parser.add_argument('--accept_sunz_min', type=int, nargs='?',
                        required=False, default=None,
                        help="Min sunz angle")
    parser.add_argument('--accept_time_diff', type=int, nargs='?',
                        required=False, default=None,
                        help="Allowd time difference in seconds")
    parser.add_argument('--max_distance_between_pixels_m', type=int, nargs='?',
                        required=False, default=None,
                        help="Allowd max distance (m)")

    options = parser.parse_args()
    n19_files = glob.glob(
        "{:s}/S_NWC_avhrr_noaa19_*T*.nc".format(options.n19_test))
    viirs_files = glob.glob(
        "{:s}/S_NWC_viirs_npp_*T*.nc".format(options.viirs_dir))
    vgac_files = glob.glob(
        "{:s}/S_NWC_avhrr_vgacsnpp_*T*.nc".format(options.vgac_dir))
    apply_network_and_plot(options, n19_files, viirs_files, vgac_files)
