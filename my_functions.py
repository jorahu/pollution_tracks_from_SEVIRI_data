#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 11:49:04 2019

@author: jormav
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.path import Path
from netCDF4 import Dataset, num2date

from scipy.signal import find_peaks

import sys
import os

import linecache
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# %%

def get_mask_by_thresh(reff_masked_):
    """ Limits Reff values by threshold and masks data """
    reff_masked_vals_ = reff_masked_[reff_masked_.nonzero()] # Only takes values >0.
    # Percentiles
    perc_pol = np.percentile(reff_masked_vals_, 25)
    perc_low = np.percentile(reff_masked_vals_, 30)
    perc_high = np.percentile(reff_masked_vals_, 55)
    
    residual_low = (perc_low-perc_pol)*1e6 # pol and unpol residual, if the values are very similar, the algorithm proceeds to look for the plateau
    residual_high = (perc_high - perc_low)*1e6 # unpol high and low residual
    tolerance = 0.015
    if residual_low <= tolerance or residual_high <= tolerance: # If plateau is present, proceed
        
        reff_vals_sorted = np.sort(reff_masked_vals_)*1e6 # Sorted r_eff values
    
        unique, counts = np.unique(reff_vals_sorted, return_counts=True) # Counts unique values and gets the counts
        most_common_idx = np.where(counts==counts.max())[0][0]
        
        border = unique[most_common_idx] # Limit value
        tol = 0.01 # Arbitrary tolerance value if the changes are very small
        reff_with_plateau = np.where(np.logical_and(reff_vals_sorted>= border-tol, reff_vals_sorted<=border+tol), -1, reff_vals_sorted) # increases the plateau, making it easier for following find_peaks function
        # Makes new counting if the areas are added due to tolerance value
        unique, counts = np.unique(reff_with_plateau, return_counts=True)
        most_common_idx = np.where(counts==counts.max())[0][0]
        if most_common_idx == 0: # Find_peaks won't work properly if the plateau starts from the beginning of data
            reff_with_plateau = np.insert(reff_with_plateau, 0, 40) # 40 is just random large value compared to data
            peaks, peak_plateaus = find_peaks( -reff_with_plateau, plateau_size = 10.)
            for key in peak_plateaus:
                peak_plateaus[key][0] -= 1
        else:
            peaks, peak_plateaus = find_peaks( -reff_with_plateau, plateau_size = 10.)

        rval = reff_vals_sorted[peak_plateaus['right_edges'][0]] # Rightmost r_eff value from plateau
        
        percs = np.arange(0, 101, 1) # Theoretical input percentile values, 1% resolution
        p = np.percentile(reff_vals_sorted, percs) # percentiles
        nearest = find_nearest(p, rval) # Finds the closest percentile value corresponding to the right edge of plateau
        nearest_max = np.max(np.where(p==nearest)) # If values are very similar then takes the highest percentile
        nearest_max += 0 # Raises the polluted limit, if necessary
        pol_high = nearest_max
        unpol_low = nearest_max + 5 # Unpolluted percentile lower limit
        unpol_high = nearest_max + 30 # Unpolluted percentile upper limit
        if unpol_high > 100:
            unpol_high = 100 # Percentile can not exceed 100%
        
        
        perc_pol = np.percentile(reff_masked_vals_, pol_high)
        perc_low = np.percentile(reff_masked_vals_, unpol_low)
        perc_high = np.percentile(reff_masked_vals_, unpol_high)
    else:
        pol_high = 25
        unpol_low = 30 # Unpol lower percentile limit
        unpol_high = 55 # Unpol upper percentile limit
        peak_plateaus = -9999.

    # Pol-unpol data and masks
    polluted_ = np.where(reff_masked_ <= perc_pol, reff_masked_, 0.)
    polluted_mask_ = np.where(polluted_ == 0., False, True)
    unpolluted_ = np.where(np.logical_and(reff_masked_ > perc_low, reff_masked_ < perc_high), reff_masked_, 0.)
    unpolluted_mask_ = np.where(unpolluted_ == 0., False, True)
    
    return polluted_mask_, unpolluted_mask_, perc_pol, perc_low, perc_high, pol_high, unpol_low, unpol_high, peak_plateaus



def filt(datafile, ts_):
    """ Initial filtering """
    x_size = datafile.dimensions['x'].size # 3712
    y_size = datafile.dimensions['y'].size # 3712
    filtr = np.ones((x_size, y_size))
    
    sunz = datafile.variables['sunz'][ts_, :, :] # Sun azimuth
    # Sun azimuth angle lower than 80 degrees
    filtr = np.where(sunz < 80. , filtr, 0.)
    
    cldmask = datafile.variables['cldmask'][ts_, :, :] # Cloudmask
    # cloudmask == 3 -> cloud filled; cloudmask == 2 -> cloud contaminated
    # Sometimes aerosol polluted clouds were classified as cloud contaminated
    filtr = np.where(np.logical_or(cldmask == 3.0, cldmask == 2.0), filtr, 0.)

    cph = datafile.variables['cph'][ts_, :, :] # Cloud thermodynamic PHase
    # cph == 1 -> liquid phase
    filtr = np.where(cph == 1.0, filtr, 0.)
    
    cth = datafile.variables['cth'][ts_, :, :] # Cloud top height
    # Cloud lower than 5000 meters
    filtr = np.where(cth < 5000, filtr, 0.)

    return filtr

# %%
def datetime_from_nc(_datafile):
    """ Gets dates from netCDF file """
    time_var = _datafile.variables['time'][:] # datetime values from file
    t_unit = _datafile.variables['time'].units # get unit  "days since 1950-01-01T00:00:00Z"
    calendar = 'standard' 
    datetime = num2date(time_var, units = t_unit, calendar = calendar) # datetime array
    return datetime

class Formatter(object):
    """ Helper function for the imshow function """
    def __init__(self, im):
        self.im = im
    def __call__(self, x, y):
        z = self.im.get_array()[int(y), int(x)]
        return 'x={:.01f}, y={:.01f}, z={:.01f}'.format(x, y, z)

def imshow(mat):
    """ Prints the value of mouse cursor at the bottom of figure obj """
    figu, axx = plt.subplots()
    im = axx.imshow(mat, origin = 'upper')
    if mat.dtype == 'bool':
        ofst = 30
        axx.set_xlim([np.nonzero(mat)[1][0]-ofst, np.nonzero(mat)[1][-1]+ofst])
        axx.set_ylim([np.nonzero(mat)[0][-1]+ofst, np.nonzero(mat)[0][1]-ofst])
    axx.format_coord = Formatter(im)
    plt.show()

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def PrintException():
    """ Custom print function for exceptions, also shows line where exception as discovered """
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))
    
def mapp(ax, x1, x2, y1, y2, lon_cities, lat_cities, city_names):
    """ Formalise Geo map and city names """
    ax.set_extent([x1, x2, y1, y2], ccrs.PlateCarree())
    ax.coastlines(resolution = '50m', linewidth = 0.5)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth = 0.5)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels = True,
                      linewidth = 0.2, color = 'black', alpha=0.5, linestyle = '--')
    gl.xlocator = mticker.FixedLocator(np.arange(-180., 180., 5.)) # pikkusjooned
    gl.ylocator = mticker.FixedLocator(np.arange(-90., 90., 5.)) # laiusjooned
    gl.xlabels_top = False # eemaldab ülevalt tickid
    gl.ylabels_right = False # eemaldab paremalt tickid
    # Linnad
    ax.plot(lon_cities, lat_cities, marker = 'x', ls = '', markersize = 14, mec = 'k', mew = 5, alpha = 0.7)
    ax.plot(lon_cities, lat_cities, marker = 'x', ls = '', markersize = 12, mec = 'w', mew = 1, alpha = 0.9)

    name_letters = [x[:3] for x in city_names] # esinimed
    for i in range(len(lon_cities)):
        ax.text(lon_cities[i] + 0.05, lat_cities[i] - 0.01, name_letters[i],
                 color = 'aqua', horizontalalignment='left', transform=ccrs.Geodetic(), alpha = 0.9, clip_on = True)
        
def get_param_by_mask(param, tstep, data, mask_pol_, mask_unpol_):
    """ Gets data from .nc and calculates mean within masked values """
    dat = data.variables[param][tstep,:,:]
    dat_pol = np.where(mask_pol_, dat.data, 0.)
    dat_pol_mean = np.mean(dat_pol[dat_pol != 0.])
    dat_unpol = np.where(mask_unpol_, dat.data, 0.)
    dat_unpol_mean = np.mean(dat_unpol[dat_unpol != 0.])

    return dat_pol, dat_unpol, dat_pol_mean, dat_unpol_mean

def timeseries_mean(param, df, ts, path, fig, fname_key, save_fig = False):
    """ Draws param (R_eff, CWP, COT) timeseries graphs and saves it to 'path' """
    M = 1e6
    k = 1e3
    fname_key = fname_key[3:]
    ax = fig.add_subplot(111)
    if param == 'reff':
        ppol = df.reff_pol[df.reff_pol.notna()] * M
        uupol = df.reff_unpol[df.reff_unpol.notna()] * M
        plt.ylabel(r'R$_{eff}$ [$\mu m$]', fontsize = 24)
        plt.title(r'R$_{eff}$ ' + fname_key, fontsize = 30)
        fname = 'a_reff_mean_{:s}_{:s}.png'
        
    elif param == 'cwp':
        ppol = df.cwp_pol[df.cwp_pol.notna()] * k
        uupol = df.cwp_unpol[df.cwp_unpol.notna()] * k
        plt.ylabel(r'CWP', fontsize = 24)
        plt.title(r'CWP [$\frac{g}{m^2}$] ' + fname_key, fontsize = 30)
        fname = 'a_cwp_mean_{:s}_{:s}.png'
    
    elif param == 'cot':
        ppol = df.cot_pol[df.cot_pol.notna()]
        uupol = df.cot_unpol[df.cot_unpol.notna()]
        plt.ylabel('COT', fontsize = 24)
        plt.title('COT {:s}'.format(fname_key), fontsize = 30)
        fname = 'a_cot_mean_{:s}_{:s}.png'
        
    elif param == 'sunz':
        sunz_data = df.sunz[df.sunz.notna()]
        plt.ylabel('SunZ', fontsize = 24)
        fname = 'a_sunz_mean_{:s}_{:s}.png'
        ax.plot(sunz_data, color = 'C3', marker = 'o', label = 'sunz')
        ax.grid(which='major', axis='x', linestyle='-')
        ax.grid(which='major', axis='y', linestyle='--')
        ax.set_xticks(sunz_data.index)
        ax.set_xticklabels(sunz_data.index, rotation = 45)
        plt.tight_layout()
        plt.legend()
        if save_fig == True:
            plt.savefig(path + fname.format(ts.strftime('%Y_%m_%d'), fname_key))
        plt.clf()
        return

    ax.plot(ppol, color = 'C3', marker = 'o', label = 'polluted')
    ax.plot(uupol, color = 'C0', marker = 'x', label = 'unpolluted')
    ax.plot((ppol - uupol), color = 'C2', marker = 's', alpha = 0.3, label = 'Difference: pol-upol')
    ax.grid(which='major', axis='x', linestyle='-')
    ax.grid(which='major', axis='y', linestyle='--')
    ax.set_xticks(ppol.index)
    ax.set_xticklabels(ppol.index, rotation = 45)
    plt.tight_layout()
    plt.legend()
    if save_fig == True:
        plt.savefig(path + fname.format(ts.strftime('%Y_%m_%d'), fname_key))
    plt.clf()

def check_dest_dir(path):
    try:
        if not os.path.exists(path):
            os.mkdir(path)
            print ("Successfully created the directory %s " % path)
    except OSError:
        print ("Creation of the directory %s failed" % path)

def get_mask_from_verts(data_path, lons_, lats_):
    """ Calculates custom area masks """
    verts_files = os.listdir(data_path)
    temps = []
    maskk_ = np.zeros(lons_.shape, dtype = bool)
    # lat-lon väärtused
    xypix = np.vstack((lons_.ravel(), lats_.ravel())).T
    for vertice_name in verts_files:
        xycrop = np.loadtxt(data_path + vertice_name) # vertices to cut areas by
        temps.append(xycrop)
        pth = Path(xycrop, closed=False)
        mask_ = pth.contains_points(xypix)
        mask_ = mask_.reshape(lons_.shape)
        maskk_ += mask_
    vertice_corners_ = np.vstack(temps)
    return maskk_, vertice_corners_
    
def get_mask_from_verts2(data_path, lons_, lats_):
    """ Calculates single custom area masks 
        Here only single area is selected and not combined together.
    """
    maskk_ = np.zeros(lons_.shape, dtype = bool)
    # lat-lon väärtused
    xypix = np.vstack((lons_.ravel(), lats_.ravel())).T
    xycrop = np.loadtxt(data_path) # Vertices to cut by
    pth = Path(xycrop, closed=False)
    mask_ = pth.contains_points(xypix)
    mask_ = mask_.reshape(lons_.shape) # lons_ can be replaced by anything with same shape
    maskk_ += mask_
    vertice_corners_ = xycrop

    return maskk_, vertice_corners_

def filled_area_check(reff_masked_, mask_):
    """ Checks area coverage and proceeds if more than 50% of pixels are filled """
    total_datapoints = np.count_nonzero(reff_masked_)
    total_maskpoints = np.count_nonzero(mask_)
    coverage = total_datapoints / total_maskpoints
    if coverage < 0.5: # vähem, kui 50% kaetus
        print('Warning! Ala katvus alla 50%: {:.2f}% '.format(coverage*100))
        return False
    else:
        print('Success! Ala katvus: {:.2f}%'.format(coverage*100))
        return True

