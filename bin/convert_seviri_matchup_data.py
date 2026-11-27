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
from sbafs_ann.create_matchup_data_lib import convert_and_save_matchupdata_seviri
from sbafs_ann.utils import DefaultSEVIRIConfigObj

if __name__ == "__main__":
    """ Train NN SBAF"""
    parser = argparse.ArgumentParser(
        description=('Script to merge matchupdata'))
    parser.add_argument('--match_dir', type=str, nargs='?',
                        required=True, default='.',
                        help="Directory with matchup files to merge.")
    parser.add_argument('--daytime', type=str, nargs='?',
                        required=False, default="day",
                        help="Target instrument")
    
    cfg = DefaultSEVIRIConfigObj()
    options = parser.parse_args()
    files = glob.glob(f"{options.match_dir}/*.nc")
    
    if options.daytime=="day":
        convert_and_save_matchupdata_seviri(cfg, files, daytime="day")
    else:    
        cfg.channel_list = cfg.channel_list_night
        cfg.accept_sunz_min = 90
        convert_and_save_matchupdata_seviri(cfg, files, daytime="night")
