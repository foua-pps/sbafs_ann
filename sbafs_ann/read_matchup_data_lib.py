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

FCI_DICT = {
    #"ch_r04": "fci_vis_04",
    #"ch_r05": "fci_vis_05",
    "ch_r06": "fci_vis_06",
    "ch_r09": "fci_vis_08",
    #"ch_rx09": "fci_vis_09",
    "ch_r13": "fci_nir_13",
    "ch_r16": "fci_nir_16",
    "ch_r22": "fci_nir_22",
    "ch_tb67": "fci_wv_63",
    "ch_tb73": "fci_wv_73",
    "ch_tb11": "fci_ir_105",
    "ch_tb12": "fci_ir_123",
    "ch_tb133": "fci_ir_133",
    "ch_tb37": "fci_ir_38",
    "ch_tb85": "fci_ir_87",
    "ch_tb97": "fci_ir_97",
    "sunzenith": "seviri_satellite_azimuth_angle",
    "satzenith": "fci_satellite_zenith_angle",
    }
SEVIRI_DICT = {
    "ch_r06": "seviri_VIS006",
    "ch_r09": "seviri_VIS008",
    "ch_tb67": "seviri_WV_062",
    "ch_tb73": "seviri_WV_073",
    "ch_r16": "seviri_IR_016",
    "ch_tb37": "seviri_IR_039",
    "ch_tb85": "seviri_IR_087",
    "ch_tb97": "seviri_IR_097",
    "ch_tb11": "seviri_IR_108",
    "ch_tb12": "seviri_IR_120",
    "ch_tb133": "seviri_IR_134",
    "sunzenith": "seviri_satellite_azimuth_angle",
    "satzenith": "seviri_satellite_zenith_angle",
    #"sun": "seviri_solar_azimuth_angle",
    #"satzenith": "seviri_solar_zenith_angle",
}




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

def get_merged_matchups_for_files_seviri_fci(cfg, files, write=False):
    seviri_obj_all = Lvl1cObj(cfg)
    viirs_obj_all = Lvl1cObj(cfg)
    tic = time.time()
    for filename in sorted(files):
        tic_i = time.time()
        print("Memory usage {:3.1f}, reading {:s}".format(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024*1024), filename))
        seviri_obj, viirs_obj = read_matchupdata_seviri(cfg, filename)
        print("Reading one file took: {:3.1f} seconds".format(time.time() - tic_i))
        viirs_obj_all += viirs_obj
        seviri_obj_all += seviri_obj
        print(seviri_obj_all.channels["ch_tb11"].shape)
        print("Merging one file took: {:3.1f} seconds".format(time.time() - tic_i))
    print("Reading ALL files took: {:3.1f} seconds".format(time.time() - tic))
    return seviri_obj_all, viirs_obj_all


def thin_and_save_matchupdata_seviri(cfg, filename):
    seviri_obj, fci_obj = read_matchupdata_seviri(cfg, filename)
    seviri_obj, fci_obj = thin_matchupdata_seviri(seviri_obj, fci_obj)
    if write:
        sub_dir = filename.split("/")[-2]
        write_matchupdata(
            "{:s}/merged_matchup_data_{:s}.h5".format(
                os.path.dirname(filename), sub_dir.lower()),
            seviri_obj, fci_obj) 
    
def read_matchupdata_seviri(cfg, filename):
    seviri_obj = Lvl1cObj(cfg, instrument="seviri")
    fci_obj = Lvl1cObj(cfg)

    seviri_var_list = [channel for channel in list(seviri_obj.channels.keys())
                       if channel in cfg.channel_list] + ["sunzenith", "satzenith"]
    with netCDF4.Dataset(filename, 'r', format='NETCDF4') as match_fh:
        for channel in cfg.channel_list + ["sunzenith", "satzenith"]:
            fci_obj.channels[channel] = match_fh[FCI_DICT[channel]][:,:].ravel()
            fci_obj.channels[channel] = np.ma.masked_array(
                fci_obj.channels[channel], mask=fci_obj.channels[channel] < 0)
            if channel in seviri_var_list:
                seviri_obj.channels[channel] = match_fh[SEVIRI_DICT[channel]][:,:].ravel()
                seviri_obj.channels[channel] = np.ma.masked_array(
                    seviri_obj.channels[channel], mask=seviri_obj.channels[channel] < 0)
        fci_obj.data["abs_time_diff_s"] = np.zeros(fci_obj.channels[channel].shape)
        fci_obj.data["distance_between_pixels_m"] = np.zeros(fci_obj.channels[channel].shape)
    cut_matched_data_according_to_cfg(cfg, seviri_obj, fci_obj)
    if write:
        sub_dir = filename.split("/")[-2]
        write_matchupdata(
            "{:s}/merged_matchup_data_{:s}.h5".format(
                os.path.dirname(filename), sub_dir.lower()),
            n19_obj_all, viirs_obj_all)
    return seviri_obj, fci_obj

if __name__ == '__main__':
    pass
