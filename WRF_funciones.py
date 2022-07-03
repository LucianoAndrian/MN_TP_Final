import glob
import os

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import warnings

warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", module="matplotlib\..*")

from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.switch_backend('agg')

import cartopy.feature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs

from wrf import getvar, interplevel, to_np, get_basemap, ll_to_xy, latlon_coords
########################################################################################################################
def PlotFields(esquema, variable, tiempos, nivel_sfc_layer=0, dominio='02',
               dpi=200, cmap='Spectral_r', escala=np.linspace(0,1,11), extend='both'):

    if (not isinstance(esquema, str)) or (not isinstance(variable, str)):
        print('esquema y variable deben ser strings')
        return

    files_dir = './tp_final/WRF_' + esquema + '/'
    out_dir = './plots/'
    #------------------------------------------------------------------------------------------------------------------#
    #Seleccion de archivos wrfout segun el domino (d01 o d02) en la carpeta del esquema indicado
    file_list = glob.glob(files_dir + 'wrfout_d' + dominio + '_*')
    file_list.sort()
    ntimes = len(file_list)
    # -----------------------------------------------------------------------------------------------------------------#
    #Si se ingresa un solo valor en tiempos
    try:
        len(tiempos)
    except:
        tiempos = [tiempos]
    # -----------------------------------------------------------------------------------------------------------------#
    #Fecha y hora para nombre y titulos
    from datetime import datetime, timedelta
    datelist = [datetime(2015, 8, 4, 0) + timedelta(hours=h) for h in range(0, 67)]
    # -----------------------------------------------------------------------------------------------------------------#


    for t in tiempos:
        ncfile = Dataset(file_list[t])
        v = getvar(ncfile, variable)

        if len(v.shape)>2:
            v = v[nivel_sfc_layer,:,:]

        if variable == 'RAINC':
            print('sumando RAINNC')
            v2 = getvar(ncfile, 'RAINNC')
            v += v2

        lats, lons = latlon_coords(v)
        lats = to_np(lats)
        lons = to_np(lons)

        #---------------------------------------------------------------------------------------------------------------
        fig = plt.figure(figsize=(5, 5), dpi=dpi)
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0))
        crs_latlon = ccrs.PlateCarree()
        ax.set_extent([-60, -57, -33, -36.5], crs_latlon)

        im = ax.contourf(lons, lats, to_np(v), levels=escala, cmap=cmap, extend=extend)
        cb = plt.colorbar(im, fraction=0.042, pad=0.035, shrink=0.8)
        cb.ax.tick_params(labelsize=8)

        ax.add_feature(cartopy.feature.LAND, facecolor='lightgrey')
        ax.add_feature(cartopy.feature.COASTLINE)
        ax.add_feature(cartopy.feature.STATES)
        ax.set_xticks(np.linspace(-60, -57, 6), crs=crs_latlon)
        ax.set_yticks(np.linspace(-33, -36, 6), crs=crs_latlon)
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        ax.tick_params(labelsize=7)
        ax.grid(color='k', alpha=0.3)

        if variable == 'RAINC':
            v_name = 'RAINC + RAINNC'
            v_name_fig = 'RAINC_RAINNC'
        else:
            v_name = v.name
            v_name_fig = v.name

        title = esquema + ': ' + v_name + ' ' + '(' + v.units + ')  ' +\
                str(datelist[t].day) + '/0' +\
                str(datelist[t].month) + '/' +\
                str(datelist[t].year) + ' ' + str(datelist[t].hour)  + 'hs'

        name_fig = v_name_fig + '_' + str(datelist[t].day) + '_0' +\
                str(datelist[t].month) + '_' + str(datelist[t].year) +\
                   '_' + str(datelist[t].hour) + '.jpg'

        plt.title(title, size=12)
        plt.tight_layout()
        plt.savefig(out_dir + esquema  + '_' + name_fig)
        plt.close('all')

# PlotFields(esquema='CLM4', variable='RAINC', tiempos=58,
#            cmap='YlGnBu', escala=np.linspace(0,100,11), extend='max')
########################################################################################################################
def PlotTimeEvolution(variable, lons_points, lats_points, dominio='02',
                      dpi=200):
    # ------------------------------------------------------------------------------------------------------------------#
    files_dir = './tp_final/WRF_CLM4/'
    files_dir2 = './tp_final/WRF_Noah-MP/'
    files_dir3 = './tp_final/WRF_RUC/'
    out_dir = './plots/'
    # -----------------------------------------------------------------------------------------------------------------#
    #Fecha y hora para nombre y titulos
    from datetime import datetime, timedelta
    datelist = [datetime(2015, 8, 4, 0) + timedelta(hours=h) for h in range(0, 67)]
    #hours = ['0'+ str(datelist[h].day) + '/' + str(datelist[h].hour) + 'hs' for h in [0,10,20,30,40,50,60]]
    #------------------------------------------------------------------------------------------------------------------#
    #Seleccion de archivos wrfout segun el domino (d01 o d02) en la carpeta del esquema indicado
    file_list = glob.glob(files_dir + 'wrfout_d' + dominio + '_*')
    file_list.sort()

    file_list2 = glob.glob(files_dir2 + 'wrfout_d' + dominio + '_*')
    file_list2.sort()

    file_list3 = glob.glob(files_dir3 + 'wrfout_d' + dominio + '_*')
    file_list3.sort()

    ntimes = len(file_list)
    ncfile = Dataset(file_list[0])

    # -----------------------------------------------------------------------------------------------------------------#
    #Si se ingresa un solo punto latxlon
    try:
        len(lons_points)
    except:
        lons_points = [lons_points]
        lats_points = [lats_points]

    num_points = range(0,len(lons_points))
    for p in num_points:
        try:
            point_lon = lons_points[p]
            point_lat = lats_points[p]

            [point_x, point_y] = ll_to_xy(ncfile, point_lat, point_lon, meta=False)
        except:
            print('Error en lons/lats points')
            return

        #topografia
        ter = getvar(ncfile, "ter")

        v1 = np.zeros(ntimes)
        v2 = np.zeros(ntimes)
        v3 = np.zeros(ntimes)
        time = np.zeros(ntimes)

        for ifile, my_file in enumerate(file_list):

            ncfile1 = Dataset(file_list[ifile])
            ncfile2 = Dataset(file_list2[ifile])
            ncfile3 = Dataset(file_list3[ifile])
            v=getvar(ncfile1, variable)

            if variable != 'RAINC':
                v1[ifile] = getvar(ncfile1, variable)[point_y, point_x]
                v2[ifile] = getvar(ncfile2, variable)[point_y, point_x]
                v3[ifile] = getvar(ncfile3, variable)[point_y, point_x]

            else:
                v1[ifile] = getvar(ncfile1, variable).mean() + getvar(ncfile1, 'RAINNC').mean()
                v2[ifile] = getvar(ncfile2, variable).mean() + getvar(ncfile2, 'RAINNC').mean()
                v3[ifile] = getvar(ncfile3, variable).mean() + getvar(ncfile3, 'RAINNC').mean()

            time[ifile] = getvar(ncfile3, 'times')
        time = (time - time[0]) / 3600.0e9
        lats, lons = latlon_coords(ter)
        lats = to_np(lats)
        lons = to_np(lons)

        fig = plt.figure(figsize=(11, 6), dpi=dpi)
        crs_latlon = ccrs.PlateCarree()
        ax1 = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=0))
        plot_ter = to_np(ter)
        plot_ter[plot_ter <= 1.0] = np.nan
        ax1.contourf(lons, lats, plot_ter, levels=np.linspace(-100, 250, 1000), cmap='terrain', extend='max')
        ax1.plot(point_lon, point_lat, 'o', color='r')
        ax1.set_extent([-60, -57, -33, -36.5], crs_latlon)
        ax1.add_feature(cartopy.feature.LAND, facecolor='lightgrey')
        ax1.add_feature(cartopy.feature.COASTLINE)
        ax1.add_feature(cartopy.feature.STATES)
        ax1.set_xticks(np.linspace(-60, -57, 6), crs=crs_latlon)
        ax1.set_yticks(np.linspace(-33, -36, 6), crs=crs_latlon)
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax1.xaxis.set_major_formatter(lon_formatter)
        ax1.yaxis.set_major_formatter(lat_formatter)
        ax1.set_title('Ubicacion del meteograma')
        ax1.grid()

        import matplotlib.dates as mdates
        divider = make_axes_locatable(ax1)
        ax2 = divider.new_horizontal(size="100%", pad=0.33, axes_class=plt.Axes)

        fig.add_axes(ax2)
        ax2.plot(datelist, v1, 'r.-', label='CLM4')
        ax2.plot(datelist, v2, 'b.-', label='Noah-MP')
        ax2.plot(datelist, v3, 'g.-', label='RUC')
        fmt = mdates.DateFormatter('%d/%Hhs')
        ax2.xaxis.set_major_formatter(fmt)
        ax2.grid()

        if variable == 'RAINC':
            v_name = 'RAINC + RAINNC'
            v_name_fig = 'RAINC_RAINNC'
        else:
            v_name = v.name
            v_name_fig = v.name

        title = v_name + ' ' + '(' + v.units + ')  '
        name_fig = v_name_fig + '_'
        ax2.set_title('Meteograma: ' + title)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(out_dir + 'TimeEvol_' + name_fig + str(point_lat) + str(point_lon) + '.jpg')
########################################################################################################################
#
# PlotTimeEvolution(variable='LH',
#                   lons_points=[-59.4],# -58.8, -59.4, -57.6],
#                   lats_points = [-34.2])#, -34.2])#, -35, -35.4])