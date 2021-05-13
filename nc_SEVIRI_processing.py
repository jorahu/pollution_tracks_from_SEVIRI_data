#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 13:06:41 2020

@author: jormav
"""
import matplotlib as mpl
#mpl.use('Agg')
#killer = False

#import sys
#f = sys.argv[1]
#destpaf = sys.argv[2]

mpl.use('Qt5Agg')
killer = True
destpaf = 'destination_path'
print(destpaf)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import os
import my_functions as mf
from my_functions import imshow
import cartopy.crs as ccrs
from timeit import default_timer as timer
import time
# %% Constant data
city_names = ['Moscow', 'Cherepovets', 'Yaroslavl', 'Kirishi', 'Kostomuksha',
          'Veliky Novgorod', 'Ryazan', 'Chagoda', 'Novomoskovsk', 'Lipetsk',
          'Stary Oskol', 'Nizhny Novgorod', 'Saratov']
lon_cities = np.array([37.617, 37.916667, 39.85, 32.016667, 30.816667,
              31.27579, 39.7, 35.333333, 38.216667, 39.6,
              37.833333, 44.0075, 46.016667])
lat_cities = np.array([55.75, 59.133333, 57.616667, 59.45, 64.683333,
              58.52099, 54.6, 59.166667, 54.083333, 52.616667,
              51.3, 56.326944, 51.533333])
M = 1e6
k = 1e3
p = 2.8e-7

# Colorbar data levels for visualisation
levelsReff = np.arange(3., 30, 1.0)
levelsCWP = np.arange(20.0, 320.0, 20.0)
levelsPrecip = np.arange(0., 11., 0.5)
# %%
if killer == True: # For testing purposes
    f = './SEVIR_OPER_R___MSGCPP__L2__20161011T000000_20161012T000000_0001.nc'

else: pass
try:
     datafile = Dataset(f,'r')
     print('Using file: {:s}'.format(f[50:]))
except Exception:
     print('Datafile error')
     mf.PrintException()
#----------------------------------------------------------------------------#
# Timesteps
dt = mf.datetime_from_nc(datafile)
#----------------------------------------------------------------------------#
# Geo coordinate matrices
lata = datafile.variables['lat'][:]
latsdat = np.where(lata.data == -999., 0., lata.data)
lona = datafile.variables['lon'][:]
lonsdat = np.where(lona.data == -999., 0., lona.data)
#----------------------------------------------------------------------------#
# Reads time_thresh files
time_thresh = pd.read_csv('./time_thresh_2.csv', index_col=0,
                          parse_dates=['thresh_alg','thresh_lopp'],
                          date_parser = pd.to_datetime)
time_thresh.index = pd.to_datetime(time_thresh.index)
#----------------------------------------------------------------------------#
# Dataframe cols and variables to insert
variables_to_df = [
        'cot', 'cwp', 'reff', 'dndv', \
        'cth', 'ctt', 'sunz', 'dcld', \
        'dcot', 'dreff', 'dcwp'
        ]

cols = [
        'cot_pol', 'cot_unpol', \
        'cwp_pol', 'cwp_unpol', \
        'reff_pol','reff_unpol', \
        'dndv_pol', 'dndv_unpol', \
        'cth_unpol', 'ctt_unpol', \
        'sunz_unpol', 'dcld_unpol', \
        'dcot_unpol', 'dreff_unpol', \
        'dcwp_unpol', 'lon_mean', \
        'lat_mean'
        ]

dict_of_df = {}

path_vert = './area_vertices2/{:s}/'.format(dt[0].strftime('%Y_%m_%d'))
vert_files = os.listdir(path_vert)
try:
    for i, vvert in enumerate(vert_files): # Creates separate DataFrame for every vertice
        vvert_ind = int(vvert[-5])
        dict_of_df['df_vert_{:d}'.format(vvert_ind)] = pd.DataFrame(columns = cols, index = dt)
    print('Dict of df-s created successfully.')
except:
    print('Problem with dataframe creation.')
# %%----------------------------------------------------------------------------#
# Destination path
path = 'C:/Users/IdaJorma/Desktop/TEST/{:s}/{:s}/'.format(destpaf, dt[0].strftime('%Y_%m_%d'))
mf.check_dest_dir(path) # Checks if path exists and creates one, if needed
#----------------------------------------------------------------------------#
# Data processing
#figg = plt.figure(figsize = (16, 12))
for ts, _dt in enumerate(dt):
    alg = timer()
    if killer == True: # For testing purposes
        if ts == 1: break
        ts = 40
        _dt = dt[ts]
        print(_dt)
    else: pass

    dt_paeva_min = time_thresh.loc[pd.to_datetime(dt[0])]['thresh_alg'].min().to_pydatetime()
    dt_paeva_max = time_thresh.loc[pd.to_datetime(dt[0])]['thresh_lopp'].max().to_pydatetime()
    # Proceeds with only good datetimes and skips the rest
    if _dt >= dt_paeva_min and _dt <= dt_paeva_max:
        pass
    else:
        continue
#----------------------------------------------------------------------------#
# Datafields from NC file
    try:
        reff = datafile.variables['reff'][ts, :, :]
        reff = np.where(reff.data == -1., 0., reff.data) # get rid of fillvalue/maskedarray
        cwp = datafile.variables['cwp'][ts, :, :]
        cwp = np.where(cwp.data == -1., 0., cwp.data)
        cot = datafile.variables['cot'][ts, :, :]
        cot = np.where(cot.data == -1., 0., cot.data)
        sunz = datafile.variables['sunz'][ts, :, :]
        sunz = np.where(sunz.data == 32767., 0., sunz.data)
        precip = datafile.variables['precip'][ts, :, :]
        precip = np.where(precip.data ==-1., 0., precip.data)
        filt = mf.filt(datafile, ts)
    except Exception:
        mf.PrintException()
        print('Problem with: ts: {:02d} | dt: {:s} Took: {:.1f}'\
              .format(ts, _dt.strftime('%Y-%m-%d %H:%M'), timer()-alg))
        continue

    # For every vertice separately
    for i, vert in enumerate(vert_files):
        vert_nr = int(vert[-5])
#        if vert_nr == 3: pass
#        else: continue
        print('Using vert: {:d} at {:s}'.format(vert_nr, _dt.strftime('%Y-%m-%d %H:%M')))

        try: # If vert does not exist, go to next vertice
            # Start and end datetimes from time_thresh.csv files
            dt_vert_alg = time_thresh[time_thresh['vert']==vert_nr].loc[pd.to_datetime(dt[0])]['thresh_alg'].to_pydatetime()
            dt_vert_lopp = time_thresh[time_thresh['vert']==vert_nr].loc[pd.to_datetime(dt[0])]['thresh_lopp'].to_pydatetime()
        except KeyError:
            continue

        if _dt >= dt_vert_alg and _dt <= dt_vert_lopp:
            pass
        else:
            continue

        maskk, vertice_corners = mf.get_mask_from_verts2(path_vert+vert, lonsdat, latsdat) # Separate masks for each area vertice

        # Coordinates - map extent by vertices
        x1 = vertice_corners[:, 0].min(); x2 = vertice_corners[:, 0].max() # logitude extent
        y1 = vertice_corners[:, 1].min(); y2 = vertice_corners[:, 1].max() # latitude extent
#----------------------------------------------------------------------------#
        # Checkpoint, if filter eats up all data, skip further processing for this timestep
        reff_filtered = np.where(filt == 1.0, reff, 0.) # Filtered data
        reff_unfiltmask = np.where(maskk, reff, np.nan) # Unfiltered data
        reff_filtmask = np.where(maskk, reff_filtered, 0.) # Masked and filtered data
#----------------------------------------------------------------------------#
        cwp_unfiltmask = np.where(maskk, cwp, np.nan)
        precip_unfiltmask = np.where(maskk, precip, np.nan)
#----------------------------------------------------------------------------#
        # Checks the data filled pixel count for the mask
        filled_area = mf.filled_area_check(reff_filtmask, maskk) # Bool
        if filled_area==False: continue # If the area filled is smaller than 50%, skips this timestep
#----------------------------------------------------------------------------#
        reff_vals = reff_filtmask[reff_filtmask.nonzero()]
#----------------------------------------------------------------------------#
        # Calculates masks by reff threshold values
        mask_pol, mask_unpol, perc_pol, perc_low, perc_high, pol_high, unpol_low, unpol_high, peak_plateaus = mf.get_mask_by_thresh(reff_filtmask)
#----------------------------------------------------------------------------#
        # Counts polluted and unpolluted mask pixels
        pol_maskpixelcount = np.count_nonzero(mask_pol)
        unpol_maskpixelcount = np.count_nonzero(mask_unpol)
#----------------------------------------------------------------------------#
        # Reff pol ja unpol
        reff_pol, reff_unpol, reff_pol_mean, reff_unpol_mean = mf.get_param_by_mask('reff', ts, datafile, mask_pol, mask_unpol)

        for var in variables_to_df:
            var_pol, var_unpol, var_pol_mean, var_unpol_mean = mf.get_param_by_mask(var, ts, datafile, mask_pol, mask_unpol)

            try: dict_of_df['df_vert_{:d}'.format(vert_nr)].loc[_dt]['{:s}_pol'.format(var)] = var_pol_mean
            except:
                mf.PrintException()
                print('Error with writing polluted data to df-s, dt: {:s}, vert: {:d}'.format(_dt.strftime('%Y-%m-%d %H:%M'), vert_nr))
            try: dict_of_df['df_vert_{:d}'.format(vert_nr)].loc[_dt]['{:s}_unpol'.format(var)] = var_unpol_mean
            except:
                mf.PrintException()
                print('Error with writing UNpolluted data to df-s, dt: {:s}, vert: {:d}'.format(_dt.strftime('%Y-%m-%d %H:%M'), vert_nr))
        dict_of_df['df_vert_{:d}'.format(vert_nr)].loc[_dt]['lat_mean'] = np.nanmean(np.where(maskk, latsdat, np.nan))
        dict_of_df['df_vert_{:d}'.format(vert_nr)].loc[_dt]['lon_mean'] = np.nanmean(np.where(maskk, lonsdat, np.nan))
#----------------------------------------------------------------------------#
        # Mask areas on map
        m_pol = np.where(mask_pol, 1., np.nan)
        m_upol = np.where(mask_unpol, 2., np.nan)
        m_full = np.where(np.logical_or(m_pol==1., m_upol==2.), np.nan, 3.)
        m_full = np.where(maskk, m_full, np.nan) # Everywhere else is nan except the mask
#        remaining_reff = np.where(m_full==3.0, reff_unfiltmask, np.nan)
#----------------------------------------------------------------------------#
        # Clipping data, for faster drawing
        xx1, xx2 = (np.nonzero(maskk)[0].min(), np.nonzero(maskk)[0].max())
        yy1, yy2 = (np.nonzero(maskk)[1].min(), np.nonzero(maskk)[1].max())

        lons = lonsdat[xx1:xx2, yy1:yy2]
        lats = latsdat[xx1:xx2, yy1:yy2]
        m_pol = m_pol[xx1:xx2, yy1:yy2]
        m_upol = m_upol[xx1:xx2, yy1:yy2]
        m_full = m_full[xx1:xx2, yy1:yy2]
        reff_unfiltmask = reff_unfiltmask[xx1:xx2, yy1:yy2]
        cwp_unfiltmask = cwp_unfiltmask[xx1:xx2, yy1:yy2]
        precip_unfiltmask = precip_unfiltmask[xx1:xx2, yy1:yy2]
#----------------------------------------------------------------------------#
        # Drawing
        title_fs = 18 # Title fontsize
        label_fs = 18 # x and y axis label fontsize
        cb_fs = 15 # Colorbar fontsize
#----------------------------------------------------------------------------#
#        time.sleep(5) # Waits for drawing functions to catch up and avoid rushing
#----------------------------------------------------------------------------#
        figg = plt.figure(figsize = (20, 10))
        gs = figg.add_gridspec(4, 4, height_ratios=[2.5, 1, 1, 1], hspace=0.16)
#----------------------------------------------------------------------------#
        # Raw data
        ax_reff = figg.add_subplot(gs[0,0], projection=ccrs.PlateCarree())
        mf.mapp(ax_reff, x1, x2, y1, y2, lon_cities, lat_cities, city_names)
        cmap=plt.get_cmap('magma_r')
        norm = mpl.colors.BoundaryNorm(levelsReff, ncolors=cmap.N)
        cs1 = ax_reff.pcolormesh(lons, lats, reff_unfiltmask*M,
                          transform = ccrs.PlateCarree(),
                          edgecolors='face',
                          norm=norm,
                          cmap=cmap)
        ax_reff.set_aspect('auto')
        plt.title('R_eff\n{:s}'.format(_dt.strftime('%Y-%m-%d %H:%M')), fontsize = title_fs)
        cb = figg.colorbar(cs1, orientation='horizontal', format='%g', pad = 0.07)
        cb.set_label(r'$R_{eff}$ $\mu m$')
#----------------------------------------------------------------------------#
        # Masks side-by-side
        ax_masks = figg.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
        mf.mapp(ax_masks, x1, x2, y1, y2, lon_cities, lat_cities, city_names)
        cmapp = mpl.colors.ListedColormap(['#ea5e46', '#53adad', '#dff299'])
        boundss = np.array([1., 2., 3., 4.])
        normm = mpl.colors.BoundaryNorm(boundss, cmapp.N)
        bin_mask = np.nansum((m_pol, m_upol, m_full), axis=0)
        bin_mask = np.where(bin_mask==0., np.nan, bin_mask)
        sbs = ax_masks.pcolormesh(lons, lats, bin_mask,
                             transform = ccrs.PlateCarree(),
                             norm = normm,
                             cmap=cmapp,)
        cb5 = figg.colorbar(sbs, ticks=boundss+0.5, orientation='horizontal', pad = 0.07)
        cb5.ax.set_xticklabels(['Polluted', 'Unpolluted', 'Excluded pixels'])
        cb5.ax.tick_params(labelsize=cb_fs-1)
        ax_masks.set_aspect('auto')
        ax_masks.set_xticklabels([])
        ax_masks.set_yticklabels([])
        plt.title('Masks; vert: {:d}\nPolpix: {:d}, Upolpix: {:d}'.format(vert_nr, pol_maskpixelcount, unpol_maskpixelcount), fontsize =title_fs)
#----------------------------------------------------------------------------#
        # CWP
        ax_cwp = figg.add_subplot(gs[0,2], projection=ccrs.PlateCarree())
        mf.mapp(ax_cwp, x1, x2, y1, y2, lon_cities, lat_cities, city_names)
        norm2 = mpl.colors.BoundaryNorm(levelsCWP, ncolors=cmap.N)
        cs2 = ax_cwp.pcolormesh(lons, lats, cwp_unfiltmask*k,
                          transform = ccrs.PlateCarree(),
                          norm=norm2,
                          cmap=cmap)
        ax_cwp.set_aspect('auto')
        plt.title('CWP\n{:s}'.format(_dt.strftime('%Y-%m-%d %H:%M')), fontsize = title_fs)
        cb2 = figg.colorbar(cs2, orientation='horizontal', format='%g', pad = 0.07)
        cb2.set_label(r'CWP $\frac{g}{m^2}$')
#----------------------------------------------------------------------------#
        # Precip
        ax_precip = figg.add_subplot(gs[0,3], projection=ccrs.PlateCarree())
        mf.mapp(ax_precip, x1, x2, y1, y2, lon_cities, lat_cities, city_names)
        norm3 = mpl.colors.BoundaryNorm(levelsPrecip, ncolors=cmap.N)
        cs3 = ax_precip.pcolormesh(lons, lats, precip_unfiltmask*p,
                          transform = ccrs.PlateCarree(),

                          norm=norm3,
                          cmap=cmap)
        ax_precip.set_aspect('auto')
        plt.title('Precip\n{:s}'.format(_dt.strftime('%Y-%m-%d %H:%M')), fontsize = title_fs)
        cb3 = figg.colorbar(cs3, orientation='horizontal', format='%g', pad = 0.07)
        cb3.set_label(r'Precip $\frac{mm}{h}$')

#----------------------------------------------------------------------------#
        # Reff statistics
        axi = figg.add_subplot(gs[1,:])

        axi.plot(reff_vals*M, c='dodgerblue', lw=0.5)
        reff_vals_sorted = np.sort(reff_vals)*M
        axi.plot(reff_vals_sorted, c='darkorange')
        if type(peak_plateaus)==dict:
            axi.plot(peak_plateaus['left_edges'][0], reff_vals_sorted[peak_plateaus['left_edges'][0]], 'x', c='C5')
            axi.plot(peak_plateaus['right_edges'][0], reff_vals_sorted[peak_plateaus['right_edges'][0]], 'x', c='C5')

        axi.axhline(perc_high*M, c='darkorchid', zorder=3, label='unpol_high = {:d} ;{:.4f}'.format(unpol_high, perc_high*M))
        axi.axhline(perc_low*M, c='mediumorchid', zorder=4, label='unpol_low = {:d} ;{:.4f}'.format(unpol_low, perc_low*M))
        axi.axhline(perc_pol*M, c='red', ls='--', zorder=5, label='pol = {:d} ;{:.4f}'.format(pol_high, perc_pol*M))
        axi.legend(loc=2)
#----------------------------------------------------------------------------#
        # delta ln(COT) / -delta ln(Reff)
        lncotpol = np.log(dict_of_df['df_vert_{:d}'.format(vert_nr)].cot_pol.values.astype(np.float))
        lncotunpol = np.log(dict_of_df['df_vert_{:d}'.format(vert_nr)].cot_unpol.values.astype(np.float))
        D_lnCOT = lncotpol-lncotunpol

        lnreffpol = np.log(dict_of_df['df_vert_{:d}'.format(vert_nr)].reff_pol.values.astype(np.float)*M)
        lnreffunpol = np.log(dict_of_df['df_vert_{:d}'.format(vert_nr)].reff_unpol.values.astype(np.float)*M)
        D_lnReff = lnreffpol-lnreffunpol
        relative = -D_lnCOT/D_lnReff

        lncwppol = np.log(dict_of_df['df_vert_{:d}'.format(vert_nr)].cwp_pol.values.astype(np.float)*M)
        lncwpunpol = np.log(dict_of_df['df_vert_{:d}'.format(vert_nr)].cwp_unpol.values.astype(np.float)*M)
        D_lnCWP = lncwppol-lncwpunpol

        xtime = dict_of_df['df_vert_{:d}'.format(vert_nr)].reff_pol.index.strftime('%H:%M')

        axi2 = figg.add_subplot(gs[2,:])
        axi2.plot(xtime, relative, c='C0', marker = 'o', ms=2, lw=1.5)
        axi2.set_ylabel(r'$-\frac{\Delta ln(COT)}{\Delta ln(R_{eff})}$')
        axi2.grid()

        axi3 = figg.add_subplot(gs[3,:])
        axi3.plot(xtime, D_lnCOT, c='C0', marker = 'o', ms=2, lw=1.5, label=r'$\Delta ln(COT)$')
        axi3.plot(xtime, D_lnReff, c='C1', marker = 'o', ms=2, lw=1.5, label=r'$\Delta ln(R_{eff})$')
        axi3.plot(xtime, D_lnCWP, c='C2', marker = 'o', ms=2, lw=1.5, label=r'$\Delta ln(CWP)$')

        axi3.legend(loc=2)
        axi3.grid()

#----------------------------------------------------------------------------#
        if killer == False:
            figg.savefig(path + 'reff_maskSBS_{:s}_vert_{:d}'.format(_dt.strftime('%y%m%d%H%M'), vert_nr))
            plt.close()
            print('Saved: reff_maskSBS_{:s}_vert_{:d}'.format(_dt.strftime('%y%m%d%H%M'), vert_nr))
        else:
            plt.show()
        print('Dt: {:s} Kulus: {:.1f}'.format(_dt.strftime('%y%m%d%H%M'), timer()-alg))

#----------------------------------------------------------------------------#
# Creates subdirectory for every separate csv file
try:
    print('Writing CSV files...')
    path_csv = path + 'csv/'
    mf.check_dest_dir(path_csv)
    for key in dict_of_df.keys():
        dict_of_df[key].to_csv(path_csv + 'csvdata_{:s}_{:s}.csv'.format(dt[0].strftime('%y_%m_%d'), key[3:]), sep=',')
        print(key + ' written')
    print('All CSV-s written successfully!')
except:
    mf.PrintException()
    print('Error with writing CSV files.')

#----------------------------------------------------------------------------#
# Timeseries graphs
if killer == False:
    path_timeseries = path + 'timeseries/'
    mf.check_dest_dir(path_timeseries)
    figg2 = plt.figure(figsize=(12, 10))
    for key in dict_of_df.keys():
        print('Timeseries plots for ' + key + ' written')
        mf.timeseries_mean('reff', dict_of_df[key], dt[0], path_timeseries, figg2, key, save_fig = True)
        mf.timeseries_mean('cwp', dict_of_df[key], dt[0], path_timeseries, figg2, key, save_fig = True)
        mf.timeseries_mean('cot', dict_of_df[key], dt[0], path_timeseries, figg2, key, save_fig = True)
#        mf.timeseries_mean('sunz', dict_of_df[key], dt[0], path_timeseries, figg2, save_fig = True)
    plt.close()
    print('All timeseries plots saved successfully!')

#----------------------------------------------------------------------------#
print('All done')


