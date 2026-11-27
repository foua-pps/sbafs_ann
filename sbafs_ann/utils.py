import numpy as np


class DefaultAVHRRConfigObj(object):
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

        
class DefaultSEVIRIConfigObj(object):
    """Object to hold config stuff."""

    def __init__(self):
        self.plotDir = None
        self.accept_satz_max = 85
        self.accept_sunz_max = 180
        self.accept_sunz_min = 0
        self.accept_time_diff = 300
        self.max_distance_between_pixels_m = 3000
        self.channel_list_night = ["ch_tb11",
                                   "ch_tb12",
                                   "ch_tb67",
                                   "ch_tb73",
                                   "ch_tb133",
                                   "ch_tb97",
                                   "ch_tb37",
                                   "ch_tb85",
                                   "sunzenith",
                                   "satzenith"]
        self.channel_list = ["ch_r06",
                             "ch_r09",
                             "ch_r16"] + self.channel_list_night


class Lvl1cObj(object):
    """Object to hold data."""

    def __init__(self, cfg, instrument="avhrr"):
        self.channel_list = cfg.channel_list
        self.channels = {"ch_r06": None,
                         "ch_r09": None,
                         "ch_tb11": None,
                         "ch_tb12": None,
                         "ch_tb37": None}
        self.target_instrument = instrument
        self.input_instrument = "viirs" 
        if instrument == "seviri":
            self.input_instrument = "fci" 
            self.channels["ch_tb67"] = None
            self.channels["ch_tb85"] = None
            self.channels["ch_tb73"] = None
            self.channels["ch_tb133"] = None
            self.channels["ch_tb97"] = None
            self.channels["ch_r16"] = None

        self.data = {}
        self.mask = None

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
                self.channels[channel] = other.channels[channel]
                continue
            elif other.channels[channel] is None:
                continue
            else:
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

    
def thin_training_data_1d(cfg, Xdata, Ydata, n_obs_to_use_per_var=100000):
    index = np.arange(Xdata.shape[0])
    selected = np.zeros(index.shape).astype(bool)
    nbins = 10
    np.random.seed(1)
    n_obs_to_use = (Xdata.shape[1]) * n_obs_to_use_per_var
    for ind in range(Xdata.shape[1]):
        var = Xdata[:, ind]
        bins = np.linspace(min(var), max(var) + 0.001, endpoint=True, num=nbins)
        for bin_i in range(nbins-1):
            use = np.logical_and(np.logical_and(var >= bins[bin_i], var < bins[bin_i+1]), ~selected)
            try:
                selection_i = np.random.choice(index[use], size=int(n_obs_to_use_per_var/nbins), replace=False)
            except:
                print("only selecting {:d}, ind {:d}, bin_i {:}".format(np.sum(use), ind, bin_i))
                selection_i = index[use]
            selected[selection_i] = True
    use = ~selected
    selection_i = np.random.choice(index[use], size=n_obs_to_use - np.sum(selected), replace=False)
    selected[selection_i] = True
    return Xdata[selected, :], Ydata[selected, :]


def thin_training_data(cfg, Xdata, Ydata,n_obs_to_use_per_var=100000):
    print(Xdata.shape)
    Xdata, Ydata = thin_training_data_1d(cfg, Xdata, Ydata, n_obs_to_use_per_var=n_obs_to_use_per_var)
    print(Xdata.shape)
    return Xdata, Ydata


def get_good_data(cfg, viirs, n19):
    mask = np.logical_or(
        n19.channels["ch_tb11"].mask, viirs.channels["ch_tb11"].mask)
    for channel in cfg.channel_list:
        if channel in n19.channels:
            mask = np.logical_or(mask, n19.channels[channel].mask)
        mask = np.logical_or(mask, viirs.channels[channel].mask)
    return ~mask

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

def remove_matched_nodata(cfg, n19_obj, viirs_obj):
    use = get_good_data(cfg, viirs_obj, n19_obj)
    select_data_according_to_use(n19_obj, viirs_obj, use)

def cut_matched_data_according_to_cfg(cfg, n19_obj, viirs_obj):
    use = get_data_to_use_cfg(cfg, n19_obj, viirs_obj)
    select_data_according_to_use(n19_obj, viirs_obj, use)


def select_data_according_to_use(n19_obj, viirs_obj, use):
    for var in n19_obj.channels:
        if n19_obj.channels[var] is not None:
            n19_obj.channels[var] = n19_obj.channels[var][use]
    for var in viirs_obj.channels:
        if viirs_obj.channels[var] is not None:
            viirs_obj.channels[var] = viirs_obj.channels[var][use]
    for var in viirs_obj.data:
        viirs_obj.data[var] = viirs_obj.data[var][use]
