from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import glob
import scipy.io as sio
import cartopy.feature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import warnings
#warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
warnings.filterwarnings('ignore')
plt.switch_backend('agg')
########################################################################################################################
esquema = 'Noah-MP'
########################################################################################################################
files_dir = './tp_final/WRF_' + esquema + '/'
out_dir = './plots/'
########################################################################################################################

from wrf import getvar, interplevel, to_np, get_basemap, latlon_coords

plot_time = 58     #En que tiempo vamos a hacer el grafico (arrancando desde 0)
figure_name = 'T_uv'   #Un nombre que distinga esta figura de otros tipos de figuras.
nivel      = 1000      #Altura del nivel (m) a donde interpolaremos los datos.
plot_mat   = True     #Si es True grafica el mapa, sino ponerlo en False

file_list = glob.glob(files_dir + 'wrfout_d02_*')   #Busco todos los wrfout en la carpeta indicada.
file_list.sort()
ntimes = len( file_list ) #Encuentro la cantidad de tiempos disponibles.

for plot_time in [17,41,65] :#range(12, len(file_list)):
    ncfile = Dataset(file_list[plot_time])

    # z = getvar(ncfile, "height",units='m')
    # [um , vm] = getvar(ncfile, "uvmet", units="m s-1")
    tk = getvar(ncfile, "LH")
    #tk2 = getvar(ncfile, "RAINNC")
    #tk = tk[0,:,:]

    # Interpolamos verticalmente a la altura seleccionada
    # um_z = interplevel(um, z, nivel)
    # vm_z = interplevel(vm, z, nivel)
    # t_z  = interplevel(tk, z, nivel)

    # Calculo la velocidad del viento
    # wspd_z = np.sqrt(um_z**2 + vm_z**2)
    t_z = tk
    # Obtenemos las lat y lons correspondientes a nuestras variables.
    lats, lons = latlon_coords(t_z)
    lats = to_np(lats)
    lons = to_np(lons)
    # Nota: Por defecto wrfpython genera variables que son objetos Xarray estos son
    # tipos de datos y metadatos. Para convertir los datos a arrays de numpy esta la
    # funcion to_np que toma el Xarray, extrae los datos como un array de numpy.

    # Creamos la figura.
    dpi = 150
    fig = plt.figure(figsize=(5, 5), dpi=dpi)
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0))
    crs_latlon = ccrs.PlateCarree()
    ax.set_extent([-60, -57, -33, -36.5], crs_latlon)

    # Graficamos la temperatura en contornos
    levels = np.arange(np.round(to_np(t_z).min()) - 2., np.round(to_np(t_z).max()) + 2., 2.)
    im = ax.contourf(lons, lats, to_np(t_z), levels=np.linspace(-350, 350, 11), cmap='RdBu_r', extend='both')
    cb = plt.colorbar(im, fraction=0.042, pad=0.035, shrink=0.8)
    cb.ax.tick_params(labelsize=8)
    ax.add_feature(cartopy.feature.LAND, facecolor='lightgrey')
    ax.add_feature(cartopy.feature.COASTLINE)
    ax.add_feature(cartopy.feature.STATES)
    #ax.gridlines(crs=crs_latlon, linewidth=0.3, linestyle='-')
    ax.set_xticks(np.linspace(-60, -57, 6), crs=crs_latlon)
    ax.set_yticks(np.linspace(-33, -36, 6), crs=crs_latlon)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.tick_params(labelsize=7)
    ax.grid(color='k', alpha=0.3)
    plt.title('TSK (ºC)' + ' h = ' + str(plot_time) + 'h')
    plt.savefig(out_dir + esquema + '_PRUEBA_' + str(plot_time) + '.png')
    plt.close('all')

#plt.show()
# Agregamos los contornos de velocidad de viento.
# levels = [1,5,10,15]
# contour=plt.contour(lons,lats,to_np(wspd_z),levels=levels,colors='k')
# plt.clabel(contour,inline=1, fontsize=10, fmt="%i")

# Agregamos el viento en barbas pero graficando solo cada 10 puntos de reticula.
skip=10
# plt.barbs(lons[::skip,::skip],lats[::skip,::skip],to_np(um_z[::skip, ::skip]),to_np(vm_z[::skip, ::skip]),length=6)


#Agregamos las gridlines
#plt.grid()
#Ajustamos los limites de la figura al dominio del WRF
#plt.axis( [ lons.min() , lons.max() , lats.min() , lats.max() ] )
#Agregamos un titulo para la figura

#Guardamos la figura en un archivo, el nombre del archivo incluye el tiempo y el nivel asi como el figure_name que definimos al principio
#plt.savefig( './' + figure_name + '_tiempo_' + str( plot_time ) + '_altura_' + str( nivel ) + '.png' )
#El plt.show es opcional solo si se desea ver la figura en tiempo real. Sino se guarda automaticamente la figura en el archivo y no se
#muestra por pantalla.
