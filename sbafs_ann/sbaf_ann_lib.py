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
                viirs_obj_all.channels[channel]
            vgac2_obj_all.channels[channel][~n19_obj_all.mask] = ytest[:, ind, 1].copy(
            )
            vgac2_obj_all.channels[channel].mask = n19_obj_all.mask
            if channel in ["ch_tb12", "ch_tb37"]:
                vgac2_obj_all.channels[channel][~n19_obj_all.mask] += ytest[:, cfg.channel_list.index("ch_tb11"), 1]
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
    do_sbaf_plots(cfg, title_end, fig_end, "SBAF-NN",
                  vgac2_obj_all, n19_obj_all)
    do_sbaf_plots(cfg, title_end, fig_end, "SBAF-VX",
                  vgac_obj_all, n19_obj_all)

    do_sbaf_plots(cfg, title_end, fig_end, "VIIRS", viirs_obj_all, n19_obj_all)


if __name__ == '__main__':
    pass
