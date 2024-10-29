#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on 2023-03-30

Copyright (c) 2023 Erik Johansson

@author:     Erik Johansson
@contact:    <erik.johansson@smhi.se>

'''

import yaml
import numpy as np
import netCDF4
from pyresample.geometry import SwathDefinition
from pyresample.kd_tree import get_neighbour_info
from pyresample.kd_tree import get_sample_from_neighbour_info
import datetime
import os


PPS_MBAND = {"ch_r06": "M05",
             "ch_r09": "M07",
             "ch_r13": "M09",
             "ch_r16": "M10",
             "ch_tb37": "M12",
             "ch_tb85": "M14",
             "ch_tb11": "M15",
             "ch_tb12": "M16"}
NOAA19_CHANNELS = ["ch_r06", "ch_r09", "ch_tb11", "ch_tb12", "ch_tb37"]


def get_cold_37_from_viirs(viirs, n19):

    update = np.logical_and(
        n19.channels["ch_tb37"].mask, ~viirs.channels["ch_tb37"].mask)
    n19.channels["ch_tb37"][update] = viirs.channels["ch_tb37"][update]
    n19.channels["ch_tb37"].mask[update] = False

def warn_get_data_to_use_cfg(cfg, viirs, n19):

    for val, var  in zip([cfg.accept_time_diff, cfg.max_distance_between_pixels_m],
                         ["abs_time_diff_s", "distance_between_pixels_m"]):
        if val > np.max(viirs.data[var]):
            print("No data filterd out due to {:s}, max diff {:3.1f}.".format(var, np.max(viirs.data[var])))
    for val, var  in zip([cfg.accept_sunz_max, cfg.accept_satz_max],
                         ["sunzenith", "satzenith"]):
        if val > np.max(viirs.channels[var]) and val > np.max(n19.channels[var]):
            print("No data filterd out due to {:s}, max viirs {:3.1f} avhrr {:3.1f}.".format(
            var, np.max(viirs.channels[var]), np.max(n19.channels[var])))
    for val, var  in zip([cfg.accept_sunz_min],
                         ["sunzenith"]):
        if val < np.min(viirs.channels[var]) and val < np.min(n19.channels[var]):
            print("No data filterd out due to {:s}, min viirs {:3.1f} avhrr {:3.1f}.".format(
                var, np.min(viirs.channels[var]), np.min(n19.channels[var])))            


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


def create_training_data(cfg, viirs, n19):
    get_cold_37_from_viirs(viirs, n19)
    warn_get_data_to_use_cfg(cfg, viirs, n19)
    use = get_data_to_use(cfg, viirs, n19)
    Xdata = np.empty((sum(use), len(cfg.channel_list)))
    n19_channels = [ch for ch in n19.channels if ch in cfg.channel_list]
    Ydata = np.empty((sum(use), len(n19_channels)))
    for ind, channel in enumerate(cfg.channel_list):
        Xdata[:, ind] = np.copy(viirs.channels[channel][use])
        if channel in n19.channels:
            Ydata[:, ind] = np.copy(n19.channels[channel][use])
        if channel in ["ch_tb12", "ch_tb37"]:
            Xdata[:, ind] -= viirs.channels["ch_tb11"][use]
            if channel in n19.channels:
                Ydata[:, ind] -= n19.channels["ch_tb11"][use]
    n19.mask = ~use
    print(Xdata.shape)
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
        "coeff_file": "{:s}/{:s}.keras".format(nn_dir, nn_pattern),
        "xmean": "{:s}/Xtrain_mean_{:s}.txt".format(nn_dir, nn_pattern),
        "xscale": "{:s}/Xtrain_scale_{:s}.txt".format(nn_dir, nn_pattern),
        "ymean": "{:s}/ytrain_mean_{:s}.txt".format(nn_dir, nn_pattern),
        "yscale": "{:s}/ytrain_scale_{:s}.txt".format(nn_dir, nn_pattern),
        "t_loss_file": "{:s}/{:s}_tloss.txt".format(nn_dir, nn_pattern),
        "v_loss_file": "{:s}/{:s}_vloss.txt".format(nn_dir, nn_pattern),
        "nn_cfg_file": "{:s}/{:s}.cfg".format(nn_dir, nn_pattern),
        "channel_list": cfg.channel_list,
        "channel_list_mband":  mband_list,
        "channel_list_mband_out": mband_list_out,
        "n_truths": len(mband_list_out)
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
    for key in ["coeff_file", "xmean", "ymean", "xscale", "yscale", "nn_cfg_file", "t_loss_file", "v_loss_file"]:
        nn_cfg_to_file[key] = os.path.basename(nn_cfg_to_file[key])
    with open(nn_cfg["nn_cfg_file"], "w") as yaml_file:
        yaml.dump(nn_cfg_to_file, yaml_file)


def read_nn_config(nn_cfg_file):
    nn_dir = os.path.dirname(nn_cfg_file)
    with open(nn_cfg_file) as y_fh:
        nn_cfg = yaml.safe_load(y_fh.read())
    for key in ["coeff_file", "xmean", "ymean", "xscale", "yscale", "nn_cfg_file", "t_loss_file", "v_loss_file"]:
        nn_cfg[key] = os.path.join(nn_dir, nn_cfg[key])
    return nn_cfg

       
def train_network_for_files(cfg, files_train, files_valid):
    from sbafs_ann.train_sbaf_nn_lib import train_network
    from sbafs_ann.create_matchup_data_lib import get_merged_matchups_for_files
    nn_cfg = set_up_nn_file_names(cfg, cfg.output_dir)
    n19_obj_all, viirs_obj_all = get_merged_matchups_for_files(cfg, files_train)
    Xtrain, ytrain = create_training_data(cfg, viirs_obj_all, n19_obj_all)
    n19_obj_all, viirs_obj_all = get_merged_matchups_for_files(cfg, files_valid)
    Xvalid, yvalid = create_training_data(cfg, viirs_obj_all, n19_obj_all)
    train_network(nn_cfg, Xtrain, ytrain, Xvalid, yvalid)

def update_cfg_with_nn_cfg(cfg, nn_cfg):
    cfg.channel_list = nn_cfg["channel_list"]
    for cfg_name in ["accept_satz_max",
                     "accept_sunz_max",
                     "accept_sunz_min",
                     "accept_time_diff",
                     "max_distance_between_pixels_m"]:
        if getattr(cfg, cfg_name) is None:
           setattr(cfg, cfg_name, nn_cfg[cfg_name])
            
           
def apply_network_and_plot(cfg, n19_files_test, npp_files, vgac_files):

    from sbafs_ann.plots_lib import do_sbaf_plots
    from sbafs_ann.train_sbaf_nn_lib import apply_network
    from sbafs_ann.create_matchup_data_lib import merge_matchup_data_for_files, Lvl1cObj
    
    nn_cfg = read_nn_config(cfg.nn_cfg_file)
    update_cfg_with_nn_cfg(cfg, nn_cfg)
    
    n19_obj_all, viirs_obj_all = merge_matchup_data_for_files(cfg, n19_files_test, npp_files)
    #n19_obj_all, vgac_obj_all = merge_matchup_data_for_files(cfg, n19_files_test, vgac_files)
    Xtest, ytest = create_training_data(cfg, viirs_obj_all, n19_obj_all)
    ytest = apply_network(
        nn_cfg,
        Xtest)

    vgac2_obj_all = Lvl1cObj(cfg)
    for ind, channel in enumerate(cfg.channel_list):
        if channel in n19_obj_all.channels and n19_obj_all.channels[channel] is not None:
            vgac2_obj_all.channels[channel] = 0 * viirs_obj_all.channels[channel]
            try:
                vgac2_obj_all.channels[channel][~n19_obj_all.mask] = ytest[:, ind, 1].copy()
            except:
                import pdb;pdb.set_trace()
            vgac2_obj_all.channels[channel].mask = n19_obj_all.mask
            if channel in ["ch_tb12", "ch_tb37"]:
                vgac2_obj_all.channels[channel][~n19_obj_all.mask] += ytest[:, cfg.channel_list.index("ch_tb11"), 1]
        vgac2_obj_all.mask = n19_obj_all.mask

    # Make same plots:
    title_end = ', SATZ < %d, SUNZ %d - %d, TD = %d sec' % (
        cfg.accept_satz_max, cfg.accept_sunz_min, cfg.accept_sunz_max, cfg.accept_time_diff)
    fig_end = 'ch{:d}_satz-{:d}_sunz_{:d}-{:d}_td-{:d}sec'.format(
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
