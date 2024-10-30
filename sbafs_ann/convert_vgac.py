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

import numpy as np
import os
from sbafs_ann.train_sbaf_nn_lib import apply_network
from sbafs_ann.sbaf_ann_lib import read_nn_config


def reorganize_data(cfg, scene):
    """Reorganize data to apply network."""
    Xdata = np.empty((np.size(scene["M15"]), len(cfg["channel_list_mband"])))
    for ind, channel in enumerate(cfg["channel_list_mband"]):
        Xdata[:, ind] = np.copy(scene[channel].values.ravel())
        if channel in ["M16", "M12"]:
            Xdata[:, ind] -= np.copy(scene["M15"].values.ravel())
    return Xdata


def rearrange_ydata(cfg, val):
    """For channel 12 and 3.7 data is trained with differences to 11µm."""
    ind_m15 = cfg["channel_list_mband_out"].index("M15")
    for ind, channel in enumerate(cfg["channel_list_mband_out"]):
        if channel in ["M16", "M12"]:
            val[:, ind, :] += val[:, ind_m15, :]

def get_error_estimate(array, ind):
    return (0.5 * np.abs(array[:, ind, 1] - array[:, ind, 0])
            + 0.5 * np.abs(array[:, ind, 1] - array[:, ind, 2]))
           
def convert_to_vgac_with_nn(scene, day_cfg_file, night_cfg_file, twilight_cfg_file=None):
    """Apply NN SBAFS to scene."""


    day_cfg = read_nn_config(day_cfg_file)
    Xdata = reorganize_data(day_cfg, scene)
    day_val = apply_network(day_cfg, Xdata)
    rearrange_ydata(day_cfg, day_val)

    night_cfg = read_nn_config(night_cfg_file)
    Xdata = reorganize_data(night_cfg, scene)
    night_val = apply_network(night_cfg, Xdata)
    rearrange_ydata(night_cfg, night_val)

    
    if twilight_cfg_file is not None:
        twilight_cfg = read_nn_config(twilight_cfg_file)
        Xdata = reorganize_data(twilight_cfg, scene)
        twilight_val = apply_network(twilight_cfg, Xdata)
        rearrange_ydata(twilight_cfg, twilight_val)

    night = scene["sunzenith"].values >= 89
    twilight = np.logical_and(scene["sunzenith"].values < 89, scene["sunzenith"].values > 80)

    ch_size = scene["M15"].values.shape
    for ind, channel in enumerate(day_cfg["channel_list_mband_out"]):
        scene[channel].values = day_val[:, ind, 1].reshape(ch_size)
        scene[channel + "_err"] = scene[channel].copy()
        scene[channel + "_err"].attrs.pop("id_tag")
        scene[channel + "_err"].attrs["add_offset"] = 0
        scene[channel + "_err"].values = get_error_estimate(day_val, ind).reshape(ch_size)

    for ind, channel in enumerate(night_cfg["channel_list_mband_out"]):
        scene[channel].values[night] = night_val[:, ind, 1].reshape(ch_size)[night] 
        scene[channel + "_err"].values[night] = get_error_estimate(night_val, ind).reshape(ch_size)[night]
        
    if twilight_cfg_file is not None:       
        for ind, channel in enumerate(twilight_cfg["channel_list_mband_out"]):
            scene[channel].values[twilight] = twilight_val[:, ind, 1].reshape(ch_size)[twilight]
            scene[channel + "_err"].values[twilight] = get_error_estimate(twilight_val, ind).reshape(ch_size)[twilight]
    return scene


if __name__ == '__main__':
    pass
