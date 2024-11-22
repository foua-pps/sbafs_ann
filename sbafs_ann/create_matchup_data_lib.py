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
COMPRESS_LVL = 6


class ConfigObj(object):
    """Object to hold config stuff."""

    def __init__(self):
        self.plotDir = None
        self.accept_satz_max = 50
        self.accept_sunz_max = 180
        self.accept_sunz_min = 0
        self.accept_time_diff = 300
        self.max_distance_between_pixels_m = 3000
        self.channel_list = ["ch_r06",
                             "ch_r09",
                             "ch_r16",
                             "ch_tb11",
                             "ch_tb12",
                             "ch_tb37",
                             "ch_tb85",
                             "sunzenith",
                             "satzenith"]


class Lvl1cObj(object):
    """Object to hold data."""

    def __init__(self, cfg):
        self.channel_list = cfg.channel_list
        self.channels = {"ch_r06": None,
                         "ch_r09": None,
                         "ch_tb11": None,
                         "ch_tb12": None,
                         "ch_tb37": None}
        self.data = {}
        self.mask = None
        self.lat = None
        self.lon = None

    def __add__(self, other):
        """Adding two objects together"""
        if self.channels["ch_tb11"] is None:
            # print("Self is None")
            return other
        if other.channels["ch_tb11"] is None:
            print("other is None")
            return self
        for channel in self.channels:
            if self.channels[channel] is None:
                continue
            self.channels[channel] = np.ma.concatenate(
                [self.channels[channel], other.channels[channel]])
            try:
                self.channels[channel].mask[0]
            except IndexError:
                self.channels[channel].mask = np.zeros(
                    self.channels[channel].shape).astype(bool)
        for dataset in self.data:
            self.data[dataset] = np.concatenate([self.data[dataset], other.data[dataset]])

        try:
            self.lat = np.concatenate([self.lat, other.lat])
            self.lon = np.concatenate([self.lon, other.lon])
        except BaseException:
            self.lat = None
            self.lon = None
        if self.mask is not None:
            self.mask = np.concatenate([self.mask, other.mask])
        return self


def get_sunz_correction(scene):  # , REFL_BANDS):
    #: Modifyed apply_sunz_correction
    #: Copyed from level1c4pps
    #: /Erik
    """Apply sun zenith angle correciton to visual channels."""
    sza = scene['sunzenith']
    mu0 = np.cos(np.radians(sza))
    scaler = 24.35 / (2 * mu0 + np.sqrt(498.5225 * mu0 * mu0 + 1))
#     for band in REFL_BANDS:
#         if band not in scene:
#             continue
#         if scene[band].attrs['sun_zenith_angle_correction_applied'] == 'False':
#             scene[band].values = scene[band].values * scaler
#             scene[band].attrs['sun_zenith_angle_correction_applied'] = 'True'
    if scaler.ndim == 3:
        scaler = scaler[0, :, :]
    return scaler


def getChannel(sati, chn, ang_masked):
    no_result = True
    for n in [0, 1, 2, 3, 4, 5, 6, 7, 20, 21, 22, 23, 24, 25, 26]:
        imageN = 'image%d' % n
        if imageN in sati.variables.keys():
            if chn == sati[imageN].id_tag:
                no_result = False
                ret = sati[imageN][0, :, :]
                #: only for visable channels
                if 'ch_r' in sati[imageN].id_tag:
                    if sati[imageN].sun_zenith_angle_correction_applied == 'False' and sati[imageN].units != "K":
                        #                         from level1c4pps import apply_sunz_correction
                        print("Making sunzenith angle correction for channel {:s}".format(chn))
                        scaler = get_sunz_correction(sati)
                        ret = ret * scaler
                if ret.mask.ndim != 0:
                    ret.mask[ang_masked] = True
                else:
                    ret.mask = ang_masked
                break
    if no_result:
        print('No result for %s' % chn)
        return None
    else:
        return ret


def getTimePerScanline(ncf):
    secFromStart = np.linspace(
        ncf['time_bnds'][:][0][0] * 24 * 3600,
        ncf['time_bnds'][:][0][1] * 24 * 3600,
        num=ncf['lon'].shape[0])
    epoch = datetime.datetime(1970, 1, 1)
    try:
        start_time_dt = datetime.datetime.strptime(
            ncf['time'].units, "days since %Y-%m-%dT%H:%M:%S.%f")
    except ValueError:
        start_time_dt = datetime.datetime.strptime(
            ncf['time'].units, "days since %Y-%m-%dT%H:%M:%S")
    start_time = start_time_dt - epoch
    scanLineTime = [start_time.total_seconds() + sec for sec in secFromStart]
    scanLineTime_dt = [start_time_dt +
                       datetime.timedelta(seconds=x) for x in secFromStart]
    return np.asarray(scanLineTime), np.array(scanLineTime_dt)


def findEdges(n19_obj, npp_obj, timeDiffMin):
    n19_time = n19_obj.time[:, 0]
    npp_time = npp_obj.time[:, 0]
    maxDiff = timeDiffMin
    minTime = np.max([n19_time[0], npp_time[0]]) - maxDiff
    maxTime = np.min([n19_time[-1], npp_time[-1]]) + maxDiff

    n19Cut = (n19_time >= minTime) & (n19_time <= maxTime)
    nppCut = (npp_time >= minTime) & (npp_time <= maxTime)

    return n19Cut, nppCut


def cutColumns(obj, edg):

    for key in obj.channels:
        if obj.channels[key] is None:
            continue
        obj.channels[key] = obj.channels[key][:, edg]
    obj.mask = obj.mask[:, edg]
    obj.lat = obj.lat[:, edg]
    obj.lon = obj.lon[:, edg]
    obj.time = obj.time[:, edg]


def cutEdges(obj, edg):

    for key in obj.channels:
        if obj.channels[key] is None:
            continue
        obj.channels[key] = obj.channels[key][edg, :]
    obj.mask = obj.mask[edg, :]
    obj.lat = obj.lat[edg, :]
    obj.lon = obj.lon[edg, :]
    obj.time = obj.time[edg, :]
    obj.time_dt = obj.time_dt[edg]


def cutMasked(obj):

    use = ~obj.mask
    for key in obj.channels:
        if obj.channels[key] is None:
            continue
        obj.channels[key] = obj.channels[key][use]
    obj.lat = obj.lat[use]
    obj.lon = obj.lon[use]
    obj.time = obj.time[use]
    obj.mask = None
    """
    for key in obj.channels:
        if obj.channels[key] is None:
            continue
        obj.channels[key] = obj.channels[key].ravel()
    obj.lat = obj.lat.ravel()
    obj.lon = obj.lon.ravel()
    obj.time = obj.time.ravel()
    obj.mask = None
    """


def read_data(fname, cfg, exclude=[]):
    print("Reading {:s}".format(fname))
    my_obj = Lvl1cObj(cfg)
    my_sat = netCDF4.Dataset(fname, 'r', format='NETCDF4')

    # Mask out data far away (2 degrees) from what we want.
    # Do not mask out bad data, as we do not want to match nodata pixel,
    # with second nearest neighbour, that might also be close enough.
    if cfg.accept_satz_max == 180:
        satza_masked = np.zeros(
            my_sat['satzenith'][0, :, :].data.shape).astype(bool)
    else:
        satza_masked = np.logical_and(
            my_sat['satzenith'][0, :, :].data <= 180,
            my_sat['satzenith'][0, :, :].data > (cfg.accept_satz_max + 2))
    if cfg.accept_sunz_max == 180:
        sunza_masked = np.zeros(
            my_sat['sunzenith'][0, :, :].data.shape).astype(bool)
    else:
        sunza_masked = np.logical_and(
            my_sat['sunzenith'][0, :, :].data <= 180,
            my_sat['sunzenith'][0, :, :].data > (cfg.accept_sunz_max + 2))
    if cfg.accept_sunz_min == 0:
        pass
    else:
        sunza_masked = np.logical_and(
            sunza_masked,
            np.logical_and(
                my_sat['sunzenith'][0, :, :].data < (cfg.accept_sunz_min - 2),
                my_sat['sunzenith'][0, :, :].data > 0))

    my_obj.mask = satza_masked | sunza_masked
    my_obj.lat = my_sat['lat'][:]
    my_obj.lon = my_sat['lon'][:]
    for channel in cfg.channel_list:
        if channel in exclude:
            continue
        val = getChannel(my_sat, channel, my_obj.mask)
        if val is not None:
            my_obj.channels[channel] = val
    my_obj.channels["satzenith"] = my_sat['satzenith'][0, :, :].data
    my_obj.channels["sunzenith"] = my_sat['sunzenith'][0, :, :].data

    scanLineTime, scanLineTime_dt = getTimePerScanline(my_sat)
    my_obj.time = np.tile(scanLineTime[:, np.newaxis], [
                          1, my_obj.lat.shape[1]])
    my_obj.time_dt = scanLineTime_dt
    my_sat.close()
    return my_obj


def calc_time_diff(rm_obj, sat_obj):
    rm_obj.data["abs_time_diff_s"] = np.where(rm_obj.time > sat_obj.time,
                                              rm_obj.time - sat_obj.time,
                                              sat_obj.time - rm_obj.time)


def mask_time_pixels(rm_obj, sat_obj, cfg):

    maybe_ok = rm_obj.time != 0
    time_diff = rm_obj.data["abs_time_diff_s"][maybe_ok]

    ok_time = np.array(
        [time_diff[ind] <= cfg.accept_time_diff for ind in range(len(time_diff))])
    if not ok_time.all():
        print("Warning some timedelta are larger than accept_time_diff, updating the mask")
        for channel in rm_obj.channels:
            if rm_obj.channels[channel] is not None:
                rm_obj.channels[channel].mask[maybe_ok][~ok_time] = True
        rm_obj.mask[maybe_ok][~ok_time] = True


def crop_nonvalid_data(n19_obj, viirs_obj):
    ok_data = viirs_obj.data["ok_data"]
    for channel in viirs_obj.channels:
        if viirs_obj.channels[channel] is None:
            continue
        viirs_obj.channels[channel] = viirs_obj.channels[channel][ok_data]
    for channel in n19_obj.channels:
        if n19_obj.channels[channel] is None:
            continue
        n19_obj.channels[channel] = n19_obj.channels[channel][ok_data]
    for varname in viirs_obj.data:
        viirs_obj.data[varname] = viirs_obj.data[varname][ok_data]
    for varname in n19_obj.data:
        n19_obj.data[varname] = n19_obj.data[varname][ok_data]
    n19_obj.lat = n19_obj.lat[ok_data]
    n19_obj.lon = n19_obj.lon[ok_data]
    viirs_obj.lat = viirs_obj.lat[ok_data]
    viirs_obj.lon = viirs_obj.lon[ok_data]


def do_matching(cfg, n19_obj, viirs_obj):
    rm_viirs_obj = Lvl1cObj(cfg)
    source_def = SwathDefinition(viirs_obj.lon, viirs_obj.lat)
    target_def = SwathDefinition(n19_obj.lon, n19_obj.lat)

    valid_in, valid_out, indices, distances = get_neighbour_info(
        source_def, target_def, radius_of_influence=cfg.max_distance_between_pixels_m, neighbours=1)

    distance_between_pixels_m = np.zeros(n19_obj.lat.shape) - 9
    distance_between_pixels_m[valid_out] = distances
    ok_data = np.logical_and(
        distance_between_pixels_m <= cfg.max_distance_between_pixels_m,
        distance_between_pixels_m > 0)
    rm_viirs_obj.data["distance_between_pixels_m"] = distance_between_pixels_m
    rm_viirs_obj.data["ok_data"] = ok_data

    for channel in viirs_obj.channels:
        if viirs_obj.channels[channel] is not None:
            rm_viirs_obj.channels[channel] = get_sample_from_neighbour_info(
                'nn', target_def.shape,
                viirs_obj.channels[channel],
                valid_in,
                valid_out,
                indices,
                fill_value=None)

    for channel in rm_viirs_obj.channels:
        if rm_viirs_obj.channels[channel] is not None:
            try:
                rm_viirs_obj.channels[channel].mask[0]
            except IndexError:
                rm_viirs_obj.channels[channel].mask = np.zeros(
                    rm_viirs_obj.channels[channel].shape).astype(bool)

    rm_viirs_obj.time = get_sample_from_neighbour_info('nn', target_def.shape,
                                                       viirs_obj.time,
                                                       valid_in,
                                                       valid_out,
                                                       indices)
    rm_viirs_obj.lon = get_sample_from_neighbour_info('nn', target_def.shape,
                                                      viirs_obj.lon,
                                                      valid_in,
                                                      valid_out,
                                                      indices)
    rm_viirs_obj.lat = get_sample_from_neighbour_info('nn', target_def.shape,
                                                      viirs_obj.lat,
                                                      valid_in,
                                                      valid_out,
                                                      indices)

    # Add lat/lon
    rm_viirs_obj.mask = rm_viirs_obj.channels["ch_tb11"].mask

    calc_time_diff(rm_viirs_obj, n19_obj)
    mask_time_pixels(rm_viirs_obj, n19_obj, cfg)
    crop_nonvalid_data(n19_obj, rm_viirs_obj)
    return rm_viirs_obj


def get_data_for_one_case(cfg, n19f, viirsf):

    n19_obj = read_data(n19f, cfg, exclude=["ch_r16"])
    viirs_obj = read_data(viirsf, cfg)
    n19_center_scanline = int(n19_obj.lat.shape[1] / 2)
    # xh = 15
    # cutColumns(n19_obj, list(
    #    range(n19_center_scanline - xh, n19_center_scanline + xh + 1)))
    n19_use, npp_use = findEdges(n19_obj, viirs_obj, cfg.accept_time_diff)
    cutEdges(n19_obj, n19_use)
    cutEdges(viirs_obj, npp_use)

    # Crop but not finely, avoid matching "distant" neighbours within sat/sun
    cutMasked(n19_obj)
    cutMasked(viirs_obj)

    if n19_obj.lat.shape[0] < 10:
        return Lvl1cObj(cfg), Lvl1cObj(cfg)

    rm_viirs_obj = do_matching(cfg, n19_obj, viirs_obj)
    cut_matched_data_according_to_cfg(cfg, n19_obj, rm_viirs_obj)

    return n19_obj, rm_viirs_obj


def get_matchups(cfg, n19f, viirsf):

    n19_start_time_s = os.path.basename(n19f).split("_")[5]
    npp_start_time_s = os.path.basename(viirsf).split("_")[5]
    n19_time_dt = datetime.datetime.strptime(
        n19_start_time_s, "%Y%m%dT%H%M%S%fZ")
    npp_time_dt = datetime.datetime.strptime(
        npp_start_time_s, "%Y%m%dT%H%M%S%fZ")
    time_diff = npp_time_dt - n19_time_dt
    if n19_time_dt > npp_time_dt:
        time_diff = n19_time_dt - npp_time_dt
    if time_diff.days > 0 or time_diff.seconds > 120 * 60:
        return None, None
    n19_obj, rm_viirs_obj = get_data_for_one_case(cfg, n19f, viirsf)
    if n19_obj.lat is None:
        return None, None
    return n19_obj, rm_viirs_obj


def get_merged_matchups_for_files(cfg, files):
    n19_obj_all = Lvl1cObj(cfg)
    viirs_obj_all = Lvl1cObj(cfg)
    tic = time.time()
    for filename in sorted(files):
        tic_i = time.time()
        print("Memory usage {:3.1f}, reading {:s}".format(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024*1024), filename))
        print("Reading one file took: {:3.1f} seconds".format(time.time() - tic_i))
        n19_obj, viirs_obj = read_matchupdata(cfg, filename)
        viirs_obj_all += viirs_obj
        n19_obj_all += n19_obj
        print(n19_obj_all.channels["ch_tb11"].shape)
        print("Merging one file took: {:3.1f} seconds".format(time.time() - tic_i))
    print("Reading ALL fileS took: {:3.1f} seconds".format(time.time() - tic))
    return n19_obj_all, viirs_obj_all


def create_matchup_data_for_files(cfg, n19_files, npp_files):
    counter = 0
    for n19f in n19_files:
        for viirsf in npp_files:
            n19_obj, viirs_obj = get_matchups(cfg, n19f, viirsf)
            if n19_obj is not None:
                counter += 1
                print(counter, os.path.basename(n19f), os.path.basename(viirsf))
                n19_start_time_s = os.path.basename(n19f).split("_")[5]
                npp_start_time_s = os.path.basename(viirsf).split("_")[5]
                write_matchupdata(
                    "{:s}/matchup_avhrr_s{:}_viirs_s{:}.h5".format(
                        cfg.output_dir,
                        n19_start_time_s,
                        npp_start_time_s),
                    n19_obj, viirs_obj)


def merge_matchup_data_for_files(cfg, n19_files, npp_files):
    n19_obj_all = Lvl1cObj(cfg)
    viirs_obj_all = Lvl1cObj(cfg)
    counter = 0
    for n19f in n19_files:
        for viirsf in npp_files:
            n19_obj, viirs_obj = get_matchups(cfg, n19f, viirsf)
            if n19_obj is not None:
                counter += 1
                print(counter, os.path.basename(n19f), os.path.basename(viirsf))
                viirs_obj_all += viirs_obj
                n19_obj_all += n19_obj
                print(n19_obj_all.channels["ch_tb11"].shape)
    return n19_obj_all, viirs_obj_all


def get_data_to_use_cfg(cfg, n19, viirs):

    use = viirs.data["abs_time_diff_s"] <= cfg.accept_time_diff
    use[viirs.data["distance_between_pixels_m"] > cfg.max_distance_between_pixels_m] = False
    use[viirs.channels["sunzenith"] > cfg.accept_sunz_max] = False
    use[viirs.channels["sunzenith"] < cfg.accept_sunz_min] = False
    use[viirs.channels["satzenith"] > cfg.accept_satz_max] = False
    use[viirs.channels["satzenith"] < 0] = False
    use[n19.channels["sunzenith"] > cfg.accept_sunz_max] = False
    use[n19.channels["sunzenith"] < cfg.accept_sunz_min] = False
    use[n19.channels["satzenith"] > cfg.accept_satz_max] = False
    use[n19.channels["satzenith"] < 0] = False
    return use


def write_matchupdata(filename, n19_obj, viirs_obj):

    with h5py.File(filename, 'w') as f:
        for name in viirs_obj.channel_list:
            print(name)
            f.create_dataset("viirs_{:s}".format(name), data=viirs_obj.channels[name],
                             compression=COMPRESS_LVL)
            if name in n19_obj.channels and n19_obj.channels[name] is not None:
                f.create_dataset("avhrr_{:s}".format(name), data=n19_obj.channels[name],
                                 compression=COMPRESS_LVL)
        for varname in ["abs_time_diff_s", "distance_between_pixels_m"]:
            f.create_dataset(varname, data=viirs_obj.data[varname],
                             compression=COMPRESS_LVL)
        f.create_dataset("avhrr_lat", data=n19_obj.lat,
                         compression=COMPRESS_LVL)
        f.create_dataset("avhrr_lon", data=n19_obj.lon,
                         compression=COMPRESS_LVL)
        f.create_dataset("viirs_lat", data=viirs_obj.lat,
                         compression=COMPRESS_LVL)
        f.create_dataset("viirs_lon", data=viirs_obj.lon,
                         compression=COMPRESS_LVL)


def cut_matched_data_according_to_cfg(cfg, n19_obj, viirs_obj):
    use = get_data_to_use_cfg(cfg, n19_obj, viirs_obj)
    for var in n19_obj.channels:
        if n19_obj.channels[var] is not None:
            n19_obj.channels[var] = n19_obj.channels[var][use]
    for var in viirs_obj.channels:
        if viirs_obj.channels[var] is not None:
            viirs_obj.channels[var] = viirs_obj.channels[var][use]
    for var in viirs_obj.data:
        viirs_obj.data[var] = viirs_obj.data[var][use]


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
                print(n19_var_list)
                n19_obj.channels[channel] = match_fh["avhrr_{:s}".format(channel)][...]
                n19_obj.channels[channel] = np.ma.masked_array(
                    n19_obj.channels[channel], mask=n19_obj.channels[channel] < 0)
        viirs_obj.data["abs_time_diff_s"] = match_fh["abs_time_diff_s"][...]
        viirs_obj.data["distance_between_pixels_m"] = match_fh["distance_between_pixels_m"][...]
    cut_matched_data_according_to_cfg(cfg, n19_obj, viirs_obj)
    return n19_obj, viirs_obj


if __name__ == '__main__':
    pass
