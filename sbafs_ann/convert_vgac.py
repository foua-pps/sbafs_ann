import numpy as np
import copy
import os
from sbafs_ann.train_sbaf_nn_lib import apply_network, tilted_loss


def convert_to_vgac_with_nn(scene, SBAF_NN_DIR):
    channel_list_day = ["M05", "M07", "M15", "M16", "M12", "M10", "M14"]
    channel_list_day_out = ["M05", "M07", "M15", "M16", "M12"]
    coeff_day = os.path.join(SBAF_NN_DIR, "ch7_SATZ_less_15_SUNZ_0_89_TD_1_min.keras")
    xscale_day = os.path.join(SBAF_NN_DIR, "Xtrain_scale_ch7_SATZ_less_15_SUNZ_0_89_TD_1_min.txt")
    xmean_day = os.path.join(SBAF_NN_DIR, "Xtrain_mean_ch7_SATZ_less_15_SUNZ_0_89_TD_1_min.txt")
    yscale_day = os.path.join(SBAF_NN_DIR, "ytrain_scale_ch7_SATZ_less_15_SUNZ_0_89_TD_1_min.txt")
    ymean_day = os.path.join(SBAF_NN_DIR, "ytrain_mean_ch7_SATZ_less_15_SUNZ_0_89_TD_1_min.txt")
    channel_list_night = ["M15", "M16", "M12", "M14"]
    channel_list_night_out =  ["M15", "M16", "M12"]
    coeff_night = os.path.join(SBAF_NN_DIR, "ch4_SATZ_less_25_SUNZ_90_180_TD_5_min.keras")
    xscale_night = os.path.join(SBAF_NN_DIR, "Xtrain_scale_ch4_SATZ_less_25_SUNZ_90_180_TD_5_min.txt")
    xmean_night = os.path.join(SBAF_NN_DIR, "Xtrain_mean_ch4_SATZ_less_25_SUNZ_90_180_TD_5_min.txt")
    yscale_night = os.path.join(SBAF_NN_DIR, "ytrain_scale_ch4_SATZ_less_25_SUNZ_90_180_TD_5_min.txt")
    ymean_night = os.path.join(SBAF_NN_DIR, "ytrain_mean_ch4_SATZ_less_25_SUNZ_90_180_TD_5_min.txt")
   
    Xdata= np.empty((np.size(scene["M15"]), len(channel_list_day)))
    for ind, channel in enumerate(channel_list_day):
        Xdata[:, ind] = np.copy(scene[channel].values.ravel())
        if channel in  ["M16", "M12"]:
            Xdata[:, ind] -= np.copy(scene["M15"].values.ravel())
    day_val = apply_network(Xdata, coeff_day, xscale_day, yscale_day, xmean_day, ymean_day, NUMBER_OF_TRUTHS=5)

    Xdata= np.empty((np.size(scene["M15"]), len(channel_list_night)))
    for ind, channel in enumerate(channel_list_night):
        Xdata[:, ind] = np.copy(scene[channel].values.ravel())
        if channel in  ["M16", "M12"]:
            Xdata[:, ind] -= np.copy(scene["M15"].values.ravel())
    night_val = apply_network(Xdata, coeff_night, xscale_night, yscale_night, xmean_night, ymean_night, NUMBER_OF_TRUTHS=3)
    
    night = scene["sunzenith"].values > 88
    for ind, channel in enumerate(channel_list_day_out):
        scene[channel].values = day_val[:, ind, 1].reshape(scene[channel].values.shape)
        if channel in  ["M16", "M12"]:
            scene[channel].values += day_val[:,channel_list_day_out.index("M15"),1].reshape(scene[channel].values.shape)
    for ind, channel in enumerate(channel_list_night_out):
        scene[channel].values[night] = night_val[:, ind, 1].reshape(scene[channel].values.shape)[night]
        if channel in  ["M16", "M12"]:
            scene[channel].values[night] += night_val[:,channel_list_night_out.index("M15"),1].reshape(scene[channel].values.shape)[night]

    del scene["M10"]
    del scene["M14"]
    return scene
        
if __name__ == '__main__':
    pass
