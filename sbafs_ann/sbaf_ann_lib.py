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


import yaml
import numpy as np
import netCDF4
from pyresample.geometry import SwathDefinition
from pyresample.kd_tree import get_neighbour_info
from pyresample.kd_tree import get_sample_from_neighbour_info
from sbafs_ann.train_sbaf_nn_lib import train_network
from sbafs_ann.create_matchup_data_lib import (get_merged_matchups_for_files,
                                               cut_matched_data_according_to_use)
from sbafs_ann import __version__
import datetime
import resource
import os
from sbafs_ann.train_sbaf_nn_lib import apply_network
from sbafs_ann.create_matchup_data_lib import merge_matchup_data_for_files, Lvl1cObj


PPS_MBAND = {"ch_r06": "M05",
             "ch_r09": "M07",
             "ch_r13": "M09",
             "ch_r16": "M10",
             "ch_tb37": "M12",
             "ch_tb85": "M14",
             "ch_tb11": "M15",
             "ch_tb12": "M16"}
NOAA19_CHANNELS = ["ch_r06", "ch_r09", "ch_tb11", "ch_tb12", "ch_tb37"]


def get_missing_37_from_viirs(viirs, n19):
    """Handle missing values in n19 data by copying from viirs."""
    update = np.logical_and(
        n19.channels["ch_tb37"].mask, ~viirs.channels["ch_tb37"].mask)
    n19.channels["ch_tb37"][update] = viirs.channels["ch_tb37"][update] - \
        viirs.channels["ch_tb11"][update] + n19.channels["ch_tb11"][update]
    n19.channels["ch_tb37"].mask[update] = False


def get_cold_37_from_viirs(viirs, n19):
    """Handle cold values in n19 data by copying from viirs."""
    update = n19.channels["ch_tb37"] < 220
    # Use VIIRS 11-3.7 for cold for missing or cold AVHRR 3.7 temperature
    n19.channels["ch_tb37"][update] = viirs.channels["ch_tb37"][update] - \
        viirs.channels["ch_tb11"][update] + n19.channels["ch_tb11"][update]
    n19.channels["ch_tb37"].mask[update] = False


def warn_get_data_to_use_cfg(cfg, viirs, n19):
    """Check if data is discarded. If data is not discarded might want to rerun matchup data."""
    might_need_rerun = False
    for val, var in zip([cfg.accept_time_diff, cfg.max_distance_between_pixels_m],
                        ["abs_time_diff_s", "distance_between_pixels_m"]):
        if val > np.max(viirs.data[var]):
            might_need_rerun = True
            print("No data filterd out due to {:s}, max diff {:3.1f}.".format(var, np.max(viirs.data[var])))
    for val, var in zip([cfg.accept_sunz_max, cfg.accept_satz_max],
                        ["sunzenith", "satzenith"]):
        if val > np.max(viirs.channels[var]) and val > np.max(n19.channels[var]):
            might_need_rerun = True
            print("No data filterd out due to {:s}, max viirs {:3.1f} avhrr {:3.1f}.".format(
                var, np.max(viirs.channels[var]), np.max(n19.channels[var])))
    for val, var in zip([cfg.accept_sunz_min],
                        ["sunzenith"]):
        if val < np.min(viirs.channels[var]) and val < np.min(n19.channels[var]):
            might_need_rerun = True
            print("No data filterd out due to {:s}, min viirs {:3.1f} avhrr {:3.1f}.".format(
                var, np.min(viirs.channels[var]), np.min(n19.channels[var])))
    if might_need_rerun:
        print("You might need to rerun matchup data!")


def get_data_to_use(cfg, viirs, n19):
    mask = np.logical_or(
        n19.channels["ch_tb11"].mask, viirs.channels["ch_tb11"].mask)
    for channel in cfg.channel_list:
        if channel in n19.channels:
            mask = np.logical_or(mask, n19.channels[channel].mask)
        mask = np.logical_or(mask, viirs.channels[channel].mask)
    use = ~mask
    use[viirs.data["abs_time_diff_s"] > cfg.accept_time_diff] = False
    use[viirs.data["distance_between_pixels_m"] > cfg.max_distance_between_pixels_m] = False
    use[viirs.channels["sunzenith"] > cfg.accept_sunz_max] = False
    use[viirs.channels["sunzenith"] < cfg.accept_sunz_min] = False
    use[viirs.channels["satzenith"] > cfg.accept_satz_max] = False
    use[n19.channels["sunzenith"] > cfg.accept_sunz_max] = False
    use[n19.channels["sunzenith"] < cfg.accept_sunz_min] = False
    use[n19.channels["satzenith"] > cfg.accept_satz_max] = False
    return use


def thin_training_data_1d(cfg, Xdata, Ydata):
    index = np.arange(Xdata.shape[0])
    selected = np.zeros(index.shape).astype(bool)
    nbins = 10
    np.random.seed(1)
    N_obs_to_use = (Xdata.shape[1]) * 100000
    for ind in range(Xdata.shape[1]):
        var = Xdata[:, ind]
        bins = np.linspace(min(var), max(var) + 0.001, endpoint=True, num=nbins)
        for bin_i in range(nbins-1):
            use = np.logical_and(np.logical_and(var >= bins[bin_i], var < bins[bin_i+1]), ~selected)
            try:
                selection_i = np.random.choice(index[use], size=10000, replace=False)
            except:
                print("only selecting {:d}, ind {:d}, bin_i {:}".format(np.sum(use), ind, bin_i))
                selection_i = index[use]
            selected[selection_i] = True
    use = ~selected
    selection_i = np.random.choice(index[use], size=N_obs_to_use - np.sum(selected), replace=False)
    selected[selection_i] = True
    return Xdata[selected, :], Ydata[selected, :]


def thin_training_data_2d(cfg, Xdata, Ydata):
    index = np.arange(Xdata.shape[0])
    selected = np.zeros(index.shape).astype(bool)
    nbins = 10
    np.random.seed(1)
    N_obs_to_use = (Xdata.shape[1]) * 100000
    ind_only_x = list(range(Ydata.shape[1], Xdata.shape[1]))
    for ind in range(Ydata.shape[1]):
        vary = Ydata[:, ind]
        varx = Xdata[:, ind]
        minv = min(varx) + min(vary)
        maxv = max(varx) + max(vary)
        bins = np.linspace(minv, maxv + 0.001, endpoint=True, num=nbins)
        for bin_i in range(nbins-1):
            use = np.logical_and(np.logical_and(vary >= bins[bin_i] - varx, vary < bins[bin_i+1]) - varx,
                                 ~selected)
            try:
                selection_i = np.random.choice(index[use], size=10000, replace=False)
            except:
                print("only selecting {:d}, ind {:d}, bin_i {:}".format(np.sum(use), ind, bin_i))
                selection_i = index[use]
            selected[selection_i] = True
    for ind in ind_only_x:
        var = Xdata[:, ind]
        bins = np.linspace(min(var), max(var) + 0.001, endpoint=True, num=nbins)
        for bin_i in range(nbins-1):
            use = np.logical_and(np.logical_and(var >= bins[bin_i], var < bins[bin_i+1]), ~selected)
            try:
                selection_i = np.random.choice(index[use], size=10000, replace=False)
            except:
                print("only selecting {:d}, ind {:d}, bin_i {:}".format(np.sum(use), ind, bin_i))
                selection_i = index[use]
            selected[selection_i] = True
    use = ~selected
    print(np.sum(selected))
    selection_i = np.random.choice(index[use], size=N_obs_to_use - np.sum(selected), replace=False)
    selected[selection_i] = True
    return Xdata[selected, :], Ydata[selected, :]


def thin_training_data(cfg, Xdata, Ydata, thin="2D"):
    print(Xdata.shape)
    if thin == "2D":
        Xdata, Ydata = thin_training_data_2d(cfg, Xdata, Ydata)
    elif thin == "1D":
        Xdata, Ydata = thin_training_data_1d(cfg, Xdata, Ydata)
    print(Xdata.shape)
    return Xdata, Ydata


def calculate_best_linear_fit(cfg, n19, viirs):
    """Print the best linear fit for the training data to file."""
    linear_fit = {}
    for channel in cfg.channel_list:
        if channel in n19.channels:
            k1, m1 = np.ma.polyfit(viirs.channels[channel], n19.channels[channel], 1)
            print("{:s} n19 = {:3.4f} * viirs + {:3.4f}".format(channel, k1, m1))
            linear_fit[channel] = ["{:3.4f}".format(k1), "{:3.4f}".format(m1), len(viirs.channels[channel])]
    with open(cfg.linear_fit_file, "w")  as linear_file:
        yaml.dump(linear_fit, linear_file)

def select_training_data(cfg, viirs, n19, update_37=False):
    get_missing_37_from_viirs(viirs, n19)
    if update_37:
        get_cold_37_from_viirs(viirs, n19)
    warn_get_data_to_use_cfg(cfg, viirs, n19)
    use = get_data_to_use(cfg, viirs, n19)
    cut_matched_data_according_to_use(cfg, n19, viirs, use)

    
def create_training_data(cfg, viirs, n19, thin=False, update_37=False):
    n_obs = len(n19.channels["ch_tb11"])
    Xdata = np.empty((n_obs, len(cfg.channel_list)))
    n19_channels = [ch for ch in n19.channels if ch in cfg.channel_list]
    Ydata = np.empty((n_obs, len(n19_channels)))
    for ind, channel in enumerate(cfg.channel_list):
        Xdata[:, ind] = np.copy(viirs.channels[channel])
        if channel in n19.channels:
            Ydata[:, ind] = np.copy(n19.channels[channel])
        if channel in ["ch_tb12", "ch_tb37"]:
            Xdata[:, ind] -= viirs.channels["ch_tb11"]
            if channel in n19.channels:
                Ydata[:, ind] -= n19.channels["ch_tb11"]
        if channel in ["ch_r09", "ch_r16"] and cfg.use_channel_quotas:
            Xdata[:, ind] = Xdata[:, ind]/viirs.channels["ch_r06"]
            if channel in n19.channels:
                Ydata[:, ind] = Ydata[:, ind] / n19.channels["ch_r06"]
    Xdata, Ydata = thin_training_data(cfg, Xdata, Ydata, thin)
    return (Xdata, Ydata)


def get_nn_name_from_cfg(cfg):
    now = datetime.datetime.utcnow().strftime("%Y%m%d")
    return 'ch{:d}_satz_max_{:d}_SUNZ_{:d}_{:d}_tdiff_{:d}_sec_{:s}'.format(
        len(cfg.channel_list),
        cfg.accept_satz_max,
        cfg.accept_sunz_min,
        cfg.accept_sunz_max,
        cfg.accept_time_diff,
        now)


def set_up_nn_file_names(cfg, nn_dir):
    nn_pattern = get_nn_name_from_cfg(cfg)
    mband_list = [PPS_MBAND[channel] for channel in cfg.channel_list]
    mband_list_out = [PPS_MBAND[channel] for channel in cfg.channel_list if channel in NOAA19_CHANNELS]
    nn_cfg = {
        "nn_pattern": nn_pattern,
        "coeff_file": "{:s}/{:s}.keras".format(nn_dir, nn_pattern),
        "xmean": "{:s}/Xtrain_mean_{:s}.txt".format(nn_dir, nn_pattern),
        "xscale": "{:s}/Xtrain_scale_{:s}.txt".format(nn_dir, nn_pattern),
        "ymean": "{:s}/ytrain_mean_{:s}.txt".format(nn_dir, nn_pattern),
        "yscale": "{:s}/ytrain_scale_{:s}.txt".format(nn_dir, nn_pattern),
        "t_loss_file": "{:s}/{:s}_tloss.txt".format(nn_dir, nn_pattern),
        "v_loss_file": "{:s}/{:s}_vloss.txt".format(nn_dir, nn_pattern),
        "nn_cfg_file": "{:s}/{:s}.yaml".format(nn_dir, nn_pattern),
        "linear_fit_file": "{:s}/{:s}_best_linear_fit.yaml".format(nn_dir, nn_pattern),
        "channel_list": cfg.channel_list,
        "channel_list_mband": mband_list,
        "channel_list_mband_out": mband_list_out,
        "n_hidden_layer_1": cfg.n_hidden_layer_1,
        "n_hidden_layer_2": cfg.n_hidden_layer_2,
        "n_hidden_layer_3": cfg.n_hidden_layer_3,
        "n_truths": len(mband_list_out),
        "sbaf_ann-version": __version__,
        "use_channel_quotas": cfg.use_channel_quotas
    }
    for cfg_name in ["accept_satz_max",
                     "accept_sunz_max",
                     "accept_sunz_min",
                     "accept_time_diff",
                     "max_distance_between_pixels_m"]:
        nn_cfg[cfg_name] = getattr(cfg, cfg_name)
    write_nn_config(nn_cfg)
    return nn_cfg


def write_nn_config(nn_cfg):
    nn_cfg_to_file = nn_cfg.copy()
    for key in ["coeff_file",
                "xmean", "ymean",
                "xscale", "yscale",
                "nn_cfg_file",
                "t_loss_file", "v_loss_file",
                "linear_fit_file"]:
        nn_cfg_to_file[key] = os.path.basename(nn_cfg_to_file[key])
    with open(nn_cfg["nn_cfg_file"], "w") as yaml_file:
        yaml.dump(nn_cfg_to_file, yaml_file)


def read_nn_config(nn_cfg_file):
    nn_dir = os.path.dirname(nn_cfg_file)
    with open(nn_cfg_file) as y_fh:
        nn_cfg = yaml.safe_load(y_fh.read())
    for key in ["coeff_file", "xmean", "ymean", "xscale", "yscale",
                "nn_cfg_file", "t_loss_file", "v_loss_file", "linear_fit_file"]:
        if key in nn_cfg:
            nn_cfg[key] = os.path.join(nn_dir, nn_cfg[key])
        else:
            nn_cfg[key] = None
    return nn_cfg

    
def train_network_for_files(cfg, files_train, files_valid):
    nn_cfg = set_up_nn_file_names(cfg, cfg.output_dir)
    update_cfg_with_nn_cfg(cfg, nn_cfg)
    n19_obj_all, viirs_obj_all = get_merged_matchups_for_files(cfg, files_train)
    select_training_data(cfg, viirs_obj_all, n19_obj_all, update_37=True)
    print("Memory usage {:3.1f}".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024*1024)))
    n19_obj_all2, viirs_obj_all2 = get_merged_matchups_for_files(cfg, files_valid)
    print("Memory usage {:3.1f}".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024*1024)))
    select_training_data(cfg, viirs_obj_all2, n19_obj_all2, update_37=True)
    calculate_best_linear_fit(cfg, n19_obj_all + n19_obj_all2, viirs_obj_all + viirs_obj_all2)

    # It 3.7µm not updated for linear fit, update it now
    # select_training_data(cfg, viirs_obj_all, n19_obj_all, update_37=True)
    # select_training_data(cfg, viirs_obj_all2, n19_obj_all2, update_37=True)
    Xtrain, ytrain = create_training_data(cfg, viirs_obj_all, n19_obj_all, thin=cfg.thin)  
    Xvalid, yvalid = create_training_data(cfg, viirs_obj_all2, n19_obj_all2, thin=cfg.thin) 
    print("Memory usage {:3.1f}".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024*1024)))
    n19_obj_all = None
    viirs_obj_all = None
    n19_obj_all2 = None
    viirs_obj_all2 = None  
    train_network(nn_cfg, Xtrain, ytrain, Xvalid, yvalid)


def update_cfg_with_nn_cfg(cfg, nn_cfg):
    cfg.channel_list = nn_cfg["channel_list"]
    for cfg_name in ["accept_satz_max",
                     "accept_sunz_max",
                     "accept_sunz_min",
                     "accept_time_diff",
                     "use_channel_quotas",
                     "linear_fit_file",
                     "max_distance_between_pixels_m"]:
        if getattr(cfg, cfg_name, None) is None:
            setattr(cfg, cfg_name, nn_cfg[cfg_name])
            print("cfg", cfg_name, nn_cfg[cfg_name])

if __name__ == '__main__':
    pass
