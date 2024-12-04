##!/usr/bin/env python
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
from sbafs_ann.create_matchup_data_lib import (ConfigObj, get_merged_matchups_for_files)


if __name__ == "__main__":
    """ Train NN SBAF"""
    parser = argparse.ArgumentParser(
        description=('Script to merge matchupdata'))
    parser.add_argument('--match_dir', type=str, nargs='?',
                        required=True, default='.',
                        help="Directory with matchup files to merge.")
    parser.add_argument('--accept_satz_max', type=int, nargs='?',
                        required=False, default=15,
                        help="Max satz angle")
    parser.add_argument('--accept_time_diff', type=int, nargs='?',
                        required=False, default=120,
                        help="Allowd time difference in seconds")
    cfg = ConfigObj()
    options = parser.parse_args()
    #cfg["accept_time_diff"] = options.accept_time_diff
    #cfg["accept_satz_max"] = options.accept_satz_max
    files = glob.glob(
        "{:s}/matchup_avhrr_*_*viirs*.h5".format(options.match_dir))
    get_merged_matchups_for_files(cfg, files, write=True)
