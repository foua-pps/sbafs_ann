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
from sbafs_ann.create_matchup_data_lib import ConfigObj, create_matchup_data_for_files

if __name__ == "__main__":
    """ Train NN SBAF"""
    parser = argparse.ArgumentParser(
        description=('Script to train ANN SBAFs'))
    parser.add_argument('--n19_dir', type=str, nargs='?',
                        required=True, default='.',
                        help="Directory with NOAA-19 training files.")
    parser.add_argument('--viirs_dir', type=str, nargs='?',
                        required=True, default='.',
                        help="Directory with all VIIRS files.")
    parser.add_argument('-o', '--output_dir', type=str, nargs='?',
                        required=False, default='.',
                        help="Output directory where to store the train network files")
    cfg = ConfigObj()
    options = parser.parse_args()
    n19_files = glob.glob(
        "{:s}/S_NWC_avhrr_noaa19_*T*.nc".format(options.n19_dir))
    viirs_files = glob.glob(
        "{:s}/S_NWC_viirs_npp_*T*.nc".format(options.viirs_dir))
    cfg.output_dir = options.output_dir
    create_matchup_data_for_files(cfg, n19_files,  viirs_files)
