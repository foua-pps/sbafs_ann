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
from create_matchup_data_lib import Lvl1cObj
import datetime
import resource
import os
import h5py
import time

def cut_matched_data_according_to_cfg(cfg, n19_obj, viirs_obj):
    use = get_data_to_use_cfg(cfg, n19_obj, viirs_obj)
    cut_matched_data_according_to_use(cfg, n19_obj, viirs_obj, use)


def cut_matched_data_according_to_use(cfg, n19_obj, viirs_obj, use):
    for var in n19_obj.channels:
        if n19_obj.channels[var] is not None:
            n19_obj.channels[var] = n19_obj.channels[var][use]
    for var in viirs_obj.channels:
        if viirs_obj.channels[var] is not None:
            viirs_obj.channels[var] = viirs_obj.channels[var][use]
    for var in viirs_obj.data:
        viirs_obj.data[var] = viirs_obj.data[var][use]

def get_merged_matchups_for_files(cfg, files, write=False):
    n19_obj_all = Lvl1cObj(cfg)
    viirs_obj_all = Lvl1cObj(cfg)
    tic = time.time()
    for filename in sorted(files):
        tic_i = time.time()
        print("Memory usage {:3.1f}, reading {:s}".format(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024*1024), filename))
        n19_obj, viirs_obj = read_matchupdata(cfg, filename)
        print("Reading one file took: {:3.1f} seconds".format(time.time() - tic_i))
        viirs_obj_all += viirs_obj
        n19_obj_all += n19_obj
        print(n19_obj_all.channels["ch_tb11"].shape)
        print("Merging one file took: {:3.1f} seconds".format(time.time() - tic_i))
    print("Reading ALL files took: {:3.1f} seconds".format(time.time() - tic))
    if write:
        sub_dir = filename.split("/")[-2]
        write_matchupdata(
            "{:s}/merged_matchup_data_{:s}.h5".format(
                os.path.dirname(filename), sub_dir.lower()),
            n19_obj_all, viirs_obj_all)
    return n19_obj_all, viirs_obj_all


def read_matchupdata(cfg, filename):
    n19_obj = Lvl1cObj(cfg)
    viirs_obj = Lvl1cObj(cfg)

    n19_var_list = [channel for channel in list(n19_obj.channels.keys())
                    if channel in cfg.channel_list] + ["sunzenith", "satzenith"]
    with h5py.File(filename, 'r') as match_fh:
        for channel in cfg.channel_list + ["sunzenith", "satzenith"]:
            viirs_obj.channels[channel] = match_fh["viirs_{:s}".format(channel)][...]
            viirs_obj.channels[channel] = np.ma.masked_array(
                viirs_obj.channels[channel], mask=viirs_obj.channels[channel] < 0)
            if channel in n19_var_list:
                n19_obj.channels[channel] = match_fh["avhrr_{:s}".format(channel)][...]
                n19_obj.channels[channel] = np.ma.masked_array(
                    n19_obj.channels[channel], mask=n19_obj.channels[channel] < 0)
        viirs_obj.data["abs_time_diff_s"] = match_fh["abs_time_diff_s"][...]
        viirs_obj.data["distance_between_pixels_m"] = match_fh["distance_between_pixels_m"][...]
    cut_matched_data_according_to_cfg(cfg, n19_obj, viirs_obj)
    return n19_obj, viirs_obj


if __name__ == '__main__':
    pass
