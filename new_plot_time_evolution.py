import matplotlib.pyplot as plt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
warnings.filterwarnings('ignore')
plt.switch_backend('agg')
import numpy as np
import matplotlib.pyplot as plt
import cartopy.feature
from netCDF4 import Dataset
from wrf import (getvar, to_np, ll_to_xy , latlon_coords  )
import glob
import scipy.io as sio
########################################################################################################################
files_dir = './tp_final/WRF_CLM4/'
files_dir2 = './tp_final/WRF_Noah-MP/'
files_dir3 = './tp_final/WRF_RUC/'
out_dir = './plots/'

variable= 'RAINC'
variable2='RAINNC'
########################################################################################################################

from wrf import getvar, interplevel, to_np, get_basemap, latlon_coords
file_list = glob.glob(files_dir + 'wrfout_d02_*')
file_list.sort()

file_list2 = glob.glob(files_dir2 + 'wrfout_d02_*')
file_list2.sort()

file_list3 = glob.glob(files_dir3 + 'wrfout_d02_*')
file_list3.sort()

ntimes = len( file_list )

ncfile = Dataset(file_list[ 0 ])

lons_points = [-59.4, -58.8, -59.4, -57.6]
lats_points = [-34.2, -34.2, -35, -35.4]
for p in [0,1,2,3]:
    point_lon = lons_points[p]
    point_lat = lats_points[p]

    # Convierto el punto de latitud y longitud en x e y de la reticula del WRF.
    [point_x, point_y] = ll_to_xy(ncfile, point_lat, point_lon, meta=False)

    ter = getvar(ncfile, "ter")  # Altura de la topografia

    v1 = np.zeros(ntimes)
    v2 = np.zeros(ntimes)
    v3 = np.zeros(ntimes)

    v11 = np.zeros(ntimes)
    v22 = np.zeros(ntimes)
    v33 = np.zeros(ntimes)

    time = np.zeros(ntimes)
    for ifile, my_file in enumerate(file_list):
        ncfile1 = Dataset(file_list[ifile])
        #v1[ifile] = getvar(ncfile1, variable)[point_y, point_x]
        v1[ifile] = getvar(ncfile1, variable).mean() +  getvar(ncfile1, variable2).mean()

        ncfile2 = Dataset(file_list2[ifile])
        #v2[ifile] = getvar(ncfile2, variable)[point_y, point_x]
        v2[ifile] = getvar(ncfile2, variable).mean() + getvar(ncfile2, variable2).mean()

        ncfile3 = Dataset(file_list3[ifile])
        #v3[ifile] = getvar(ncfile3, variable)[point_y, point_x]
        v3[ifile] = getvar(ncfile3, variable).mean() + getvar(ncfile3, variable2).mean()

        # ncfile1 = Dataset(file_list[ ifile ])
        # v11[ ifile ] = getvar(ncfile1, 'RAINNC' )[point_y,point_x] + getvar(ncfile1, 'RAINC' )[point_y,point_x]
        #
        # ncfile2 = Dataset(file_list2[ ifile ])
        # v22[ ifile ] = getvar(ncfile2, 'RAINNC' )[point_y,point_x] + getvar(ncfile2, 'RAINC' )[point_y,point_x]
        #
        # ncfile3 = Dataset(file_list3[ ifile ])
        # v33[ ifile ] = getvar(ncfile3, 'RAINNC' )[point_y,point_x] + getvar(ncfile3, 'RAINC' )[point_y,point_x]

        time[ifile] = getvar(ncfile3, 'times')

    time = (time - time[0]) / 3600.0e9  # Pongo el tiempo en horas desde el inicio de la simulacion.

    # Obtenemos las matrices de latitud y longitud para los graficos.
    lats, lons = latlon_coords(ter)
    lats = to_np(lats)
    lons = to_np(lons)

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig = plt.figure(figsize=(10, 6), dpi=100)
    crs_latlon = ccrs.PlateCarree()
    ax1 = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=0))
    plot_ter = to_np(ter)
    plot_ter[plot_ter <= 1.0] = np.nan
    ax1.contourf(lons, lats, plot_ter, levels=np.linspace(-100, 300, 1000), cmap='terrain', extend='max')
    ax1.plot(point_lon, point_lat, 'o', color='r')
    ax1.set_extent([-60, -57, -33, -36.5], crs_latlon)
    ax1.add_feature(cartopy.feature.LAND, facecolor='lightgrey')
    ax1.add_feature(cartopy.feature.COASTLINE)
    ax1.add_feature(cartopy.feature.STATES)
    # ax1.gridlines(crs=crs_latlon, linewidth=0.3, linestyle='-', alpha=1)
    ax1.set_xticks(np.linspace(-60, -57, 6), crs=crs_latlon)
    ax1.set_yticks(np.linspace(-33, -36, 6), crs=crs_latlon)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax1.xaxis.set_major_formatter(lon_formatter)
    ax1.yaxis.set_major_formatter(lat_formatter)
    ax1.set_title('Ubicacion del meteograma')
    ax1.grid()

    divider = make_axes_locatable(ax1)
    ax2 = divider.new_horizontal(size="100%", pad=0.33, axes_class=plt.Axes)
    fig.add_axes(ax2)
    ax2.plot(time, v1, 'r.-', label='CLM4')
    ax2.plot(time, v2, 'b.-', label='Noah-MP')
    ax2.plot(time, v3, 'g.-', label='RUC')
    ax2.grid()
    ax2.set_title('Meteograma LH (W m-2)')
    ax2.legend()
    # plt.plot( time , td2m,

    plt.tight_layout()
    plt.savefig(out_dir + 'TimeEvol_Prueba_' + variable + str(point_lat) + str(point_lon) + '.png')




# divider = make_axes_locatable(ax1)
# ax3 = divider.new_horizontal(size="100%", pad=0.33, axes_class=plt.Axes)
# fig.add_axes(ax3)
# ax3.plot( time , v11 , 'r.-' ,label='CLM4')
# ax3.plot( time , v22 , 'b.-' ,label='Noah-MP')
# ax3.plot( time , v33 , 'g.-' ,label='RUC')
# ax3.grid()
# ax3.set_title('Meteograma PP acum (mm)')
# ax3.legend()
#
# plt.tight_layout()
# plt.savefig(out_dir + 'pic' + '.png')
