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
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import copy
import os

matplotlib.rcParams["font.size"] = 20


def plotScatterHisto(
        x_flat_in,
        y_flat_in,
        stitle,
        satn,
        xylim,
        pt_int,
        pt_str,
        pt_loc,
        histd2_bins,
        vmax,
        figname,
        mark_zero=False):
    inds = np.where(~(x_flat_in.mask | y_flat_in.mask))
    x_flat = x_flat_in.copy()[inds]
    y_flat = y_flat_in.copy()[inds]
    print("Number of data points {:d}".format(len(x_flat)))
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(stitle)
    ax = fig.add_subplot(1, 1, 1)
    # ax.scatter(x_flat, y_flat, marker='.')

    # xymax = np.max([np.max(x_flat), np.max(y_flat)])
    # xymin = np.min([np.min(x_flat), np.min(y_flat)])
    # binsize = (xymax-xymin)/histd2_bins
    n_edges = histd2_bins
    edgesx = np.linspace(xylim[0], xylim[1], n_edges)
    edgesy = np.linspace(xylim[0], xylim[1], n_edges)
    H, xe, ye = np.histogram2d(x_flat, y_flat, bins=[edgesx, edgesy])
    # - edgesx[0])/(n_edges+1)).astype(np.int64)
    xi = np.searchsorted(edgesx, x_flat)
    # np.floor((y - edgesy[0])/(n_edges+1)).astype(np.int64)  # -1?
    yi = np.searchsorted(edgesy, y_flat)
    xi = xi - 1
    yi = yi - 1
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Move pixels out side in side ?:
    yi[yi == n_edges] = n_edges - 1
    xi[xi == n_edges] = n_edges - 1
    # yi(yi==2)==1 This is what we need!?
    yi[yi == n_edges - 1] = n_edges - 2
    xi[xi == n_edges - 1] = n_edges - 2
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    z = H[xi, yi]
    idx = z.argsort()
    my_cmap = copy.copy(plt.get_cmap("inferno_r", lut=100))
    cmap_vals = my_cmap(np.arange(100))  # extra ctvalues as an array
    # print cmap_vals[0]
    cmap_vals[0:5] = cmap_vals[5]  # change the first values to less white
    my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "new_inferno_r", cmap_vals)
    b = ax.scatter(x_flat[idx], y_flat[idx], c=z[idx],
                   cmap=my_cmap, vmin=1, vmax=vmax, linewidth=0.5,
                   alpha=1.0, marker='.', edgecolors=None, rasterized=True)
    ax.set_title('Scatter and Density')
    ax.set_ylabel('NOAA 19, max = %d' % int(y_flat.max()))
    ax.set_xlabel('%s, max = %d' % (satn, int(x_flat.max())))
    ax.set_xlim(xylim)
    ax.set_ylim(xylim)
    ax.set_xticks(pt_int)
    ax.set_yticks(pt_int)
    # cax = fig.add_axes([0.80, 0.65, 0.06, 0.22])
    cbar = fig.colorbar(b)

    xn = np.linspace(xylim[0], xylim[1], 100)
    k1, m1 = np.ma.polyfit(x_flat, y_flat, 1)
    k2a, k2b, m2 = np.ma.polyfit(x_flat, y_flat, 2)
    # p1 = np.ma.polyfit(x_flat, y_flat, 1)
    # p2 = np.ma.polyfit(x_flat, y_flat, 2)
    ny_y1 = k1 * xn + m1
    ny_y2 = k2a * np.square(xn) + k2b * xn + m2
    MSE = np.square(y_flat - x_flat).mean()
    MAE = np.abs(y_flat - x_flat).mean()
    PE1 = 100 * np.sum(np.abs(y_flat - x_flat) > 1) / len(y_flat)
    RMSE = np.sqrt(MSE)
    ax.plot(xn, xn, 'r--', lw=1.0,
            label='y=x RMSE={:3.3f} MAE = {:3.3f} PE1= {:3.1f} N={:d}'.format(RMSE, MAE, PE1, len(x_flat)), )
    if mark_zero:
        ax.plot(xylim, [0, 0], 'k--', lw=0.5)
        ax.plot([0, 0], xylim, 'k--', lw=0.5)
    ax.plot(xn, ny_y1, 'g', label='%.4G*x%+.2f' % (k1, m1), linewidth=1.0)
    ax.plot(xn, ny_y2, 'b', label='%.4G*x²+%.4G*x%+.2f' % (k2a, k2b, m2), linewidth=1.0)
    rr = np.ma.corrcoef(x_flat, y_flat)

    ax.text(1, 0, 'ccof = %.4f' %
            rr[0, 1], horizontalalignment='right', transform=ax.transAxes)
    ax.legend(loc=2)
    ax.set_aspect(1)
    fig.savefig(figname + '.png')


def do_sbaf_plots(cfg, title_end, fig_end, what, vgac_obj_all, n19_obj_all):
    PLOT_DIR = cfg.plot_dir
    r_max_axis = 180
    tb_min_axis = 170
    tb_max_axis = 350
    vmax = 1000

    r_plot_ticks_int = [*range(0, r_max_axis + 1, 30)]
    r_plot_ticks_str = np.asarray(r_plot_ticks_int).astype(str).tolist()
    r_plot_ticks_loc = []
    tb_plot_ticks_int = [*range(tb_min_axis, tb_max_axis + 1, 30)]
    tb_plot_ticks_str = np.asarray(tb_plot_ticks_int).astype(str).tolist()
    tb_plot_ticks_loc = []
    histd2_bins = 400
    for i in range(len(r_plot_ticks_int)):
        #: -1 because we want the in between values
        r_plot_ticks_loc.append(
            int(np.round(i / (len(r_plot_ticks_int) - 1) * histd2_bins)))
    for i in range(len(tb_plot_ticks_int)):
        tb_plot_ticks_loc.append(
            int(np.round(i / (len(tb_plot_ticks_int) - 1) * histd2_bins)))
    #: Change last place to histd2_bins -1 instead of histd2_bins
    #: Othervise the figure looks vierd
    r_plot_ticks_loc[-1] = histd2_bins - 1
    tb_plot_ticks_loc[-1] = histd2_bins - 1

    # if accept_satz_max == 180:
    #    title_end = title_end.replace(', SATZ < 180', ', SATZ < inf')
    # if accept_sunz_max == 180:
    #    title_end = title_end.replace(', SUNZ < 180', ', SATZ < inf')

    if cfg.accept_sunz_min < 90:
        for channel in ["ch_r06", "ch_r09"]:
            if vgac_obj_all.channels[channel] is None:
                continue
            #: r06 simulted from VGAC
            figname = '{:s}/{:s}_n19_{:s}_{:s}'.format(
                PLOT_DIR, what, channel, fig_end)
            the_title = 'Channel ({:s}) {:s}'.format(channel[-3:], title_end)
            plotScatterHisto(vgac_obj_all.channels[channel], n19_obj_all.channels[channel], the_title, what, [0, r_max_axis],
                             r_plot_ticks_int, r_plot_ticks_str, r_plot_ticks_loc, histd2_bins, vmax, figname)

    for channel in ["ch_tb11", "ch_tb12", "ch_tb37"]:
        if vgac_obj_all.channels[channel] is None:
            continue
        #: VGAC
        figname = '{:s}/{:s}_n19_{:s}_{:s}'.format(
            PLOT_DIR, what, channel, fig_end)
        the_title = 'Channel ({:s}) {:s}'.format(channel[-4:], title_end)
        plotScatterHisto(
            vgac_obj_all.channels[channel],
            n19_obj_all.channels[channel],
            the_title,
            what,
            [
                tb_min_axis,
                tb_max_axis],
            tb_plot_ticks_int,
            tb_plot_ticks_str,
            tb_plot_ticks_loc,
            histd2_bins,
            vmax,
            figname)
    plt.close("all")
    #: tb difference t11t12
    n19_t11t12_all_flat = n19_obj_all.channels["ch_tb11"] - \
        n19_obj_all.channels["ch_tb12"]
    vgac_t11t12_all_flat = vgac_obj_all.channels["ch_tb11"] - \
        vgac_obj_all.channels["ch_tb12"]

    #: tb difference t11t37
    n19_t11t37_all_flat = n19_obj_all.channels["ch_tb11"] - \
        n19_obj_all.channels["ch_tb37"]
    vgac_t11t37_all_flat = vgac_obj_all.channels["ch_tb11"] - \
        vgac_obj_all.channels["ch_tb37"]

    #: Rearrange scaling and other things
    tb_min_axis = -3
    tb_max_axis = 9
    tb_plot_ticks_int = [*range(tb_min_axis, tb_max_axis + 1, 2)]
    tb_plot_ticks_str = np.asarray(tb_plot_ticks_int).astype(str).tolist()
    tb_plot_ticks_loc = []
    histd2_bins = 400

    for i in range(len(tb_plot_ticks_int)):
        tb_plot_ticks_loc.append(
            int(np.round(i / (len(tb_plot_ticks_int) - 1) * histd2_bins)))
    #: Change last place to histd2_bins -1 instead of histd2_bins
    #: Othervise the figure looks vierd
    tb_plot_ticks_loc[-1] = histd2_bins - 1

    #: t11t12 simulated from VGAC
    figname = '{:s}/{:s}_n19_t11t12_{:s}'.format(PLOT_DIR, what, fig_end)
    the_title = 't11t12 diff{:s}'.format(title_end)
    plotScatterHisto(vgac_t11t12_all_flat,
                     n19_t11t12_all_flat,
                     the_title,
                     what,
                     [tb_min_axis,
                      tb_max_axis],
                     tb_plot_ticks_int,
                     tb_plot_ticks_str,
                     tb_plot_ticks_loc,
                     histd2_bins,
                     vmax,
                     figname,
                     mark_zero=True)

    #: Rearrange scaling and other things
    tb_max_axis = 7
    tb_min_axis = -7
    tb_plot_ticks_int = [*range(tb_min_axis, tb_max_axis + 1, 2)]
    if np.min(n19_t11t37_all_flat) < -20:
        tb_min_axis = -50
        tb_plot_ticks_int = [*range(tb_min_axis, tb_max_axis + 1, 5)]
    tb_plot_ticks_str = np.asarray(tb_plot_ticks_int).astype(str).tolist()
    tb_plot_ticks_loc = []
    histd2_bins = 400

    for i in range(len(tb_plot_ticks_int)):
        tb_plot_ticks_loc.append(
            int(np.round(i / (len(tb_plot_ticks_int) - 1) * histd2_bins)))
    #: Change last place to histd2_bins -1 instead of histd2_bins
    #: Othervise the figure looks vierd
    tb_plot_ticks_loc[-1] = histd2_bins - 1

    #: t11t37 simulated from VGAC
    figname = '{:s}/{:s}_n19_t11t37_{:s}'.format(PLOT_DIR, what, fig_end)
    the_title = 't11t37 diff{:s}'.format(title_end)
    plotScatterHisto(vgac_t11t37_all_flat,
                     n19_t11t37_all_flat,
                     the_title,
                     what,
                     [tb_min_axis,
                      tb_max_axis],
                     tb_plot_ticks_int,
                     tb_plot_ticks_str,
                     tb_plot_ticks_loc,
                     histd2_bins,
                     vmax,
                     figname,
                     mark_zero=True)


def plot_orbit(vgac_obj, n19_obj, figname):
    npp_center_scanline = int(vgac_obj.lat.shape[1] / 2)
    n19_center_scanline = int(n19_obj.lat.shape[1] / 2)
    myFmt = mdates.DateFormatter('%H:%M')
    fig = plt.figure()
    fig.suptitle('Noaa 19 & NPP')
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(vgac_obj.time_dt,
             vgac_obj.lat[:, npp_center_scanline], label='NPP')
    ax1.plot(n19_obj.time_dt,
             n19_obj.lat[:, n19_center_scanline], label='Noaa 19')
    ax1.set_title('Org orbit')
    ax1.set_ylabel('Latitude')
    ax1.set_xlabel('Time')
    ax1.xaxis.set_major_formatter(myFmt)
    ax1.legend()
    fig.savefig(figname + '.png')


def plot_latlon(n19_obj, viirs_obj, title_end, figname):
    fig = plt.figure()
    fig.suptitle('Noaa 19 = ' + os.path.basename(n19f).split('_')[-2].replace(
        'Z', '') + '\n' + 'NPP = ' + os.path.basename(viirsf).split('_')[-2])
    ax1 = fig.add_subplot(1, 1, 1)
    if n19_obj.mask is not None:
        use = ~n19_obj.mask
        ax1.plot(n19_obj.lon[use], n19_obj.lat[use], '.b', alpha=0.1)
        print("Number of data points n19 ll masked {:d}".format(
            len(n19_obj.lon[use])))
    else:
        ax1.plot(n19_obj.lon, n19_obj.lat, '.b', alpha=0.1)
        print("Number of data points n19 ll {:d}".format(len(n19_obj.lon)))
    if viirs_obj is not None:
        if viirs_obj.mask is not None:
            use = ~viirs_obj.mask
            ax1.plot(viirs_obj.lon[use], viirs_obj.lat[use], '.r', alpha=0.1)
            print("Number of data points ll {:d}".format(
                len(viirs_obj.lon[use])))
        else:
            ax1.plot(viirs_obj.lon, viirs_obj.lat, '.r', alpha=0.1)
            print("Number of data points ll {:d}".format(len(viirs_obj.lon)))
    ax1.set_xlim([-180, 180])
    ax1.set_ylim([-90, 90])
    ax1.set_title('Geolocation' + title_end)
    ax1.set_ylabel('Latitude')
    ax1.set_xlabel('Longitude')
    fig.savefig(figname + 'png')


if __name__ == '__main__':
    pass
