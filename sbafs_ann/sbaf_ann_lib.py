#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on 2023-03-30

Copyright (c) 2023 Erik Johansson

@author:     Erik Johansson
@contact:    <erik.johansson@smhi.se>

'''

import numpy as np
import netCDF4
from pyresample.geometry import SwathDefinition
from pyresample.kd_tree import get_neighbour_info
from pyresample.kd_tree import get_sample_from_neighbour_info
import datetime
import os


class ConfigObj(object):
    """Object to hold config stuff."""

    def __init__(self):
        self.plotDir = None
        self.accept_satz_max = None
        self.accept_sunz_max = None
        self.accept_sunz_min = None
        self.accept_time_diff = None
        self.max_distance_between_pixels_m = None


class Lvl1cObj(object):
    """Object to hold data."""

    def __init__(self, cfg):
        self.channel_list = cfg.channel_list
        self.channels = {"ch_r06": None,
                         "ch_r09": None,
                         "ch_tb11": None,
                         "ch_tb12": None,
                         "ch_tb37": None}
        self.mask = None
        self.lat = None
        self.lon = None

    def __add__(self, other):
        """Adding two objects together"""
        if self.lat is None:
            # print("Self is None")
            return other
        if other.lat is None:
            print("other is None")
            return self
        for channel in self.channel_list:
            if channel not in self.channels:
                continue
            self.channels[channel] = np.ma.concatenate(
                [self.channels[channel], other.channels[channel]])
            try:
                self.channels[channel].mask[0]
            except IndexError:
                self.channels[channel].mask = np.zeros(
                    self.channels[channel].shape).astype(bool)
        self.lat = np.concatenate([self.lat, other.lat])
        self.lon = np.concatenate([self.lon, other.lon])
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
                    if sati[imageN].sun_zenith_angle_correction_applied == 'False':
                        #                         from level1c4pps import apply_sunz_correction
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
    maxDiff = timeDiffMin * 60
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


def read_data(fname, cfg, exclude=[]):
    my_obj = Lvl1cObj(cfg)
    my_sat = netCDF4.Dataset(fname, 'r', format='NETCDF4')
    if cfg.accept_satz_max == 180:
        satza_masked = np.zeros(
            my_sat['satzenith'][0, :, :].data.shape).astype(bool)
    else:
        satza_masked = my_sat['satzenith'][0, :, :].data > cfg.accept_satz_max
    if cfg.accept_sunz_max == 180:
        sunza_masked = np.zeros(
            my_sat['sunzenith'][0, :, :].data.shape).astype(bool)
    else:
        sunza_masked = my_sat['sunzenith'][0, :, :].data > cfg.accept_sunz_max
    if cfg.accept_sunz_min == 0:
        pass
    else:
        sunza_masked = np.logical_or(
            sunza_masked, my_sat['sunzenith'][0, :, :].data < cfg.accept_sunz_min)
    my_obj.mask = satza_masked | sunza_masked
    my_obj.lat = my_sat['lat'][:]
    my_obj.lon = my_sat['lon'][:]
    for channel in cfg.channel_list:
        if channel in exclude:
            continue
        val = getChannel(my_sat, channel, my_obj.mask)
        if val is not None:
            my_obj.channels[channel] = val
    scanLineTime, scanLineTime_dt = getTimePerScanline(my_sat)
    my_obj.time = np.tile(scanLineTime[:, np.newaxis], [
                          1, my_obj.lat.shape[1]])
    my_obj.time_dt = scanLineTime_dt
    my_sat.close()
    return my_obj


def mask_time_pixels(rm_obj, sat_obj, cfg):

    maybe_ok = rm_obj.time != 0
    time_diff = np.where(rm_obj.time[maybe_ok] > sat_obj.time[maybe_ok],
                         rm_obj.time[maybe_ok] - sat_obj.time[maybe_ok],
                         sat_obj.time[maybe_ok] - rm_obj.time[maybe_ok])

    ok_time = np.array(
        [time_diff[ind] < cfg.accept_time_diff * 60 for ind in range(len(time_diff))])
    if not ok_time.all():
        print("Warning some timedelta are larger than accept_time_diff, updating the mask")
        for channel in rm_obj.channels:
            if rm_obj.channels[channel] is not None:
                rm_obj.channels[channel].mask[maybe_ok][~ok_time] = True
        rm_obj.mask[maybe_ok][~ok_time] = True


def do_matching(cfg, n19_obj, viirs_obj):
    rm_viirs_obj = Lvl1cObj(cfg)
    source_def = SwathDefinition(viirs_obj.lon, viirs_obj.lat)
    target_def = SwathDefinition(n19_obj.lon, n19_obj.lat)

    valid_in, valid_out, indices, distances = get_neighbour_info(
        source_def, target_def, radius_of_influence=cfg.max_distance_between_pixels_m, neighbours=1)

    for channel in cfg.channel_list:
        if channel in viirs_obj.channels:
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
    # Add lat/lon
    rm_viirs_obj.lat = n19_obj.lat
    rm_viirs_obj.lon = n19_obj.lon
    rm_viirs_obj.mask = rm_viirs_obj.channels["ch_tb11"].mask

    mask_time_pixels(rm_viirs_obj, n19_obj, cfg)
    return rm_viirs_obj


def get_data_for_one_case(cfg, n19f, viirsf):

    n19_obj = read_data(n19f, cfg, exclude=["ch_r16"])
    viirs_obj = read_data(viirsf, cfg)

    n19_center_scanline = int(n19_obj.lat.shape[1] / 2)
    xh = 2
    cutColumns(n19_obj, list(
        range(n19_center_scanline - xh, n19_center_scanline + xh + 1)))
    n19_use, npp_use = findEdges(n19_obj, viirs_obj, cfg.accept_time_diff)
    cutEdges(n19_obj, n19_use)
    cutEdges(viirs_obj, npp_use)

    cutMasked(n19_obj)
    cutMasked(viirs_obj)

    if n19_obj.lat.shape[0] < 10:
        return Lvl1cObj(cfg), Lvl1cObj(cfg)

    rm_viirs_obj = do_matching(cfg, n19_obj, viirs_obj)
    return n19_obj, rm_viirs_obj


def get_cold_37_from_viirs(viirs, n19):
    update = np.logical_and(
        n19.channels["ch_tb37"].mask, ~viirs.channels["ch_tb37"].mask)
    n19.channels["ch_tb37"][update] = viirs.channels["ch_tb37"][update]
    n19.channels["ch_tb37"].mask[update] = False


def create_training_data(cfg, viirs, n19):
    get_cold_37_from_viirs(viirs, n19)
    mask = np.logical_or(
        n19.channels["ch_tb11"].mask, viirs.channels["ch_tb11"].mask)
    for channel in cfg.channel_list:
        if channel in n19.channels:
            mask = np.logical_or(mask, n19.channels[channel].mask)
        mask = np.logical_or(mask, viirs.channels[channel].mask)
    use = ~mask
    Xdata = np.empty((sum(use), len(cfg.channel_list)))
    n19_channels = [ch for ch in n19.channels if n19.channels[ch] is not None]
    Ydata = np.empty((sum(use), len(n19_channels)))
    for ind, channel in enumerate(cfg.channel_list):
        Xdata[:, ind] = np.copy(viirs.channels[channel][use])
        if channel in n19.channels:
            Ydata[:, ind] = np.copy(n19.channels[channel][use])
        if channel in ["ch_tb12", "ch_tb37"]:
            Xdata[:, ind] -= viirs.channels["ch_tb11"][use]
            if channel in n19.channels:
                Ydata[:, ind] -= n19.channels["ch_tb11"][use]
    n19.mask = mask

    if np.isnan(Ydata).any():
        import pdb
        pdb.set_trace()
    return (Xdata, Ydata)


def get_matchups(cfg, n19_files, npp_files):
    n19_obj_all = Lvl1cObj(cfg)
    viirs_obj_all = Lvl1cObj(cfg)
    for n19f in n19_files:
        counter = 0
        n19_start_time_s = os.path.basename(n19f).split("_")[5]
        for viirsf in npp_files:
            npp_start_time_s = os.path.basename(viirsf).split("_")[5]
            n19_time_dt = datetime.datetime.strptime(
                n19_start_time_s, "%Y%m%dT%H%M%S%fZ")
            npp_time_dt = datetime.datetime.strptime(
                npp_start_time_s, "%Y%m%dT%H%M%S%fZ")
            time_diff = npp_time_dt - n19_time_dt
            if n19_time_dt > npp_time_dt:
                time_diff = n19_time_dt - npp_time_dt

            if time_diff.days > 0 or time_diff.seconds > 120 * 60:
                continue
            n19_obj, rm_viirs_obj = get_data_for_one_case(cfg, n19f, viirsf)
            if n19_obj.lat is None:
                continue
            counter += 1
            print(counter, os.path.basename(n19f), os.path.basename(viirsf))
            # Append this orbit to the ones we already have
            n19_obj_all += n19_obj
            viirs_obj_all += rm_viirs_obj

    return n19_obj_all, viirs_obj_all


def get_nn_name_from_cfg(cfg):
    return 'ch{:d}_SATZ_less_{:d}_SUNZ_{:d}_{:d}_TD_{:d}_min'.format(
        len(
            cfg.channel_list),
        cfg.accept_satz_max,
        cfg.accept_sunz_min,
        cfg.accept_sunz_max,
        cfg.accept_time_diff)


def train_network_for_files(cfg, n19_files_train, n19_files_valid, npp_files):
    from sbafs_ann.train_sbaf_nn_lib import train_network
    nn_name = get_nn_name_from_cfg(cfg)
    n19_obj_all, viirs_obj_all = get_matchups(cfg, n19_files_train, npp_files)
    Xtrain, ytrain = create_training_data(cfg, viirs_obj_all, n19_obj_all)
    n19_obj_all, viirs_obj_all = get_matchups(cfg, n19_files_valid, npp_files)
    Xvalid, yvalid = create_training_data(cfg, viirs_obj_all, n19_obj_all)
    train_network(Xtrain, ytrain, Xvalid, yvalid,
                  NN_NAME=nn_name, OUTPUT_DIR=cfg.output_dir)


def apply_network_and_plot(cfg, n19_files_test, npp_files, vgac_files):

    from sbafs_ann.plots_lib import do_sbaf_plots
    from sbafs_ann.train_sbaf_nn_lib import apply_network_nn_name
    nn_name = get_nn_name_from_cfg(cfg)
    n19_obj_all, viirs_obj_all = get_matchups(cfg, n19_files_test, npp_files)
    n19_obj_all, vgac_obj_all = get_matchups(cfg, n19_files_test, vgac_files)
    Xtest, ytest = create_training_data(cfg, viirs_obj_all, n19_obj_all)
    ytest = apply_network_nn_name(
        Xtest,
        NN_NAME=nn_name,
        NUMBER_OF_TRUTHS=ytest.shape[1],
        OUTPUT_DIR=cfg.nn_dir)

    vgac2_obj_all = Lvl1cObj(cfg)
    for ind, channel in enumerate(cfg.channel_list):
        if channel in n19_obj_all.channels and n19_obj_all.channels[channel] is not None:
            vgac2_obj_all.channels[channel] = 0 * \
                vgac_obj_all.channels[channel]
            vgac2_obj_all.channels[channel][~n19_obj_all.mask] = ytest[:, ind, 1].copy(
            )
            vgac2_obj_all.channels[channel].mask = n19_obj_all.mask
            if channel in ["ch_tb12", "ch_tb37"]:
                vgac2_obj_all.channels[channel][~n19_obj_all.mask] += ytest[:,
                                                                            cfg.channel_list.index("ch_tb11"), 1]
        vgac2_obj_all.mask = n19_obj_all.mask

    # Make same plots:
    title_end = ', SATZ < %d, SUNZ %d - %d, TD = %d min' % (
        cfg.accept_satz_max, cfg.accept_sunz_min, cfg.accept_sunz_max, cfg.accept_time_diff)
    fig_end = 'ch{:d}_satz-{:d}_sunz_{:d}-{:d}_td-{:d}min'.format(
        len(
            cfg.channel_list),
        cfg.accept_satz_max,
        cfg.accept_sunz_min,
        cfg.accept_sunz_max,
        cfg.accept_time_diff)
    do_sbaf_plots(cfg, title_end, fig_end, "SBAF-VX",
                  vgac_obj_all, n19_obj_all)
    do_sbaf_plots(cfg, title_end, fig_end, "SBAF-NN",
                  vgac2_obj_all, n19_obj_all)
    do_sbaf_plots(cfg, title_end, fig_end, "VIIRS", viirs_obj_all, n19_obj_all)


if __name__ == '__main__':
    pass
