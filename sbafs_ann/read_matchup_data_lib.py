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


import numpy as np
import netCDF4
from pyresample.geometry import SwathDefinition
from pyresample.kd_tree import get_neighbour_info
from pyresample.kd_tree import get_sample_from_neighbour_info
import datetime
import resource
import os
import h5py
import time
from sbafs_ann.utils import Lvl1cObj, cut_matched_data_according_to_cfg
from sbafs_ann.utils import thin_training_data, remove_matched_nodata
from sbafs_ann.create_matchup_data_lib import write_matchupdata


def get_merged_matchups_for_files(cfg, files, write=False):
    n19_obj_all = Lvl1cObj(cfg)
    viirs_obj_all = Lvl1cObj(cfg)
    tic = time.time()
    for filename in sorted(files):
        tic_i = time.time()
        print("Memory usage {:3.1f}, reading {:s}".format(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024*1024), filename))
        n19_obj, viirs_obj = read_and_cut_matchupdata(cfg, filename)
        print("Reading one file took: {:3.1f} seconds".format(time.time() - tic_i))
        viirs_obj_all += viirs_obj
        n19_obj_all += n19_obj
        print(n19_obj_all.channels["ch_tb11"].shape)
        print("Merging one file took: {:3.1f} seconds".format(time.time() - tic_i))
    print("Reading ALL files took: {:3.1f} seconds".format(time.time() - tic))
    if write:
        sub_dir = filename.split("/")[-2]
        daytime = ""
        if "night" in os.path.basename(filename):
            daytime = "_night"
        if "day" in os.path.basename(filename):
            daytime = "_day"             
        fout = f"{os.path.dirname(filename)}/merged_matchup{daytime}_{sub_dir}.h5"
        write_matchupdata(fout, n19_obj_all, viirs_obj_all)
    return n19_obj_all, viirs_obj_all

def read_and_cut_matchupdata(cfg, filename):
    n19_obj, viirs_obj = read_matchupdata(cfg, filename)
    cut_matched_data_according_to_cfg(cfg, n19_obj, viirs_obj)
    return n19_obj, viirs_obj

def read_matchupdata(cfg, filename):

    with h5py.File(filename, 'r') as match_fh:

        target_instrument = match_fh.attrs["target_instrument"]
        input_instrument = match_fh.attrs["input_instrument"]
        n19_obj = Lvl1cObj(cfg, instrument=target_instrument)
        viirs_obj = Lvl1cObj(cfg, instrument=target_instrument)

        n19_var_list = [channel for channel in list(n19_obj.channels.keys())
                        if channel in cfg.channel_list] + ["sunzenith", "satzenith"]
    
        for channel in cfg.channel_list + ["sunzenith", "satzenith"]:
            var_name = f"{input_instrument}_{channel}"
            viirs_obj.channels[channel] = match_fh[var_name][...]
            viirs_obj.channels[channel] = np.ma.masked_array(viirs_obj.channels[channel],
                                                             mask=viirs_obj.channels[channel] < 0)
            if channel in n19_var_list:
                var_name = f"{target_instrument}_{channel}"                                
                n19_obj.channels[channel] = match_fh[var_name][...]
                n19_obj.channels[channel] = np.ma.masked_array(
                    n19_obj.channels[channel], mask=n19_obj.channels[channel] < 0)
        viirs_obj.data["abs_time_diff_s"] = match_fh["abs_time_diff_s"][...]
        viirs_obj.data["distance_between_pixels_m"] = match_fh["distance_between_pixels_m"][...]
    return n19_obj, viirs_obj


def get_filename_out(filename):
    return f"{os.path.dirname(filename)}/merged_matchup_data_{sub_dir.lower()}.h5"




if __name__ == '__main__':
    pass
