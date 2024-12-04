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
                        required=False, default=5*60,
                        help="Allowd time difference in seconds")
    parser.add_argument('--max_distance_between_pixels_m', type=int, nargs='?',
                        required=False, default=3000,
                        help="Allowd max distance (m)")
    parser.add_argument('--thinning_method_2D', const=True, nargs='?',
                        required=False,
                        help="Use 2D thinning of data for training")
    parser.add_argument('-o', '--output_dir', type=str, nargs='?',
                        required=False, default='.',
                        help="Output directory where to store the train network files")
    parser.add_argument('--n_hidden_layer_1', type=int, nargs='?',
                        required=False, default=15,
                        help="Number of nodes in first hidden layer")
    parser.add_argument('--n_hidden_layer_2', type=int, nargs='?',
                        required=False, default=15,
                        help="Number of nodes in first hidden layer")
    parser.add_argument('--n_hidden_layer_3', type=int, nargs='?',
                        required=False, default=0,
                        help="Number of nodes in first hidden layer")
    parser.add_argument('--use_channel_quotas', type=bool, nargs='?',
                        required=False, default=True,
                        help="Use channel quotas")
              
    options = parser.parse_args()
    files_train = glob.glob(
        "{:s}/matchup_avhrr_*_*viirs*.h5".format(options.train_dir))
    files_valid = glob.glob(
        "{:s}/matchup_avhrr_*_*viirs*.h5".format(options.valid_dir))
    if len(files_train)<1 or len(files_valid)<1:
        raise ValueError("Missing training or/and validation files!")
    options.thin = "1D"
    if options.thinning_method_2D:
        options.thin = "2D"
    
    train_network_for_files(options, files_train, files_valid)
