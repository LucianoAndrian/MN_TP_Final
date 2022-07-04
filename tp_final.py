from WRF_funciones import PlotFields, PlotTimeEvolution
import numpy as np
########################################################################################################################
"""
Variables directas afectadas por el esquema de suelo: TSK, LH. SMOIS
variables indeirectas: PLBH, PP
"""
#----------------------------------------------------------------------------------------------------------------------#
# Ploteo de todos los campos de las variables de arriba
# solo PBLH seran usados en la presentacion en forma de gif (dpi=200)
# el resto para tener mejor idea de la situacion (dpi=50)

escalas = [np.linspace(0,350,8), np.linspace(5,25,11), np.linspace(0,1,11),
           np.linspace(0,1000,11), np.linspace(0,100,11)]
colorbar = ['gist_heat_r', 'Spectral_r', 'YlGnBu', 'Spectral_r', 'YlGnBu']
extend = ['both', 'both', 'min','max', 'max']

for esquema in ['CLM4', 'Noah-MP', 'RUC']:
    v_count=0
    for variable in ['LH', 'TSK', 'SMOIS', 'PBLH', 'RAINC']:
        if variable == 'PBLH':
            dpi=200
        else:
            dpi=50
        PlotFields(esquema, variable, dpi=dpi,
                   escala=escalas[v_count],
                   cmap=colorbar[v_count],
                   extend=extend[v_count],
                   tiempos=range(0,67))
        v_count +=1
#----------------------------------------------------------------------------------------------------------------------#
# Evolucion temporal de sobre el dominio para pp
PlotTimeEvolution('RAINC', dominio_total=True, lons_points=-666, lats_points=-666)

# Evolucion temporal de sobre el dominio para SMOIS. WOW
PlotTimeEvolution('SMOIS', dominio_total=True, lons_points=-666, lats_points=-666,
                  nivel_sfc_layer=0)

# Evolucion temporal de sobre el dominio para TSK. WOW
PlotTimeEvolution('TSK', dominio_total=True, lons_points=-666, lats_points=-666)

#----------------------------------------------------------------------------------------------------------------------#
# Evolucion temporal sobre el dominio de las cuencas de los rios Areco y Arrecifes
PlotTimeEvolution('RAINC',sub_dominio=True,
                  sub_lons=[-60,-59.1], sub_lats=[-34.35,-33.7],
                  lons_points=-666, lats_points=-666)
#SMOIS
PlotTimeEvolution('SMOIS',sub_dominio=True,nivel_sfc_layer=0,
                  sub_lons=[-60,-59.1], sub_lats=[-34.35,-33.7],
                  lons_points=-666, lats_points=-666)
#TSK
PlotTimeEvolution('TSK',sub_dominio=True,
                  sub_lons=[-60,-59.1], sub_lats=[-34.35,-33.7],
                  lons_points=-666, lats_points=-666)
#LH
PlotTimeEvolution('LH',sub_dominio=True,
                  sub_lons=[-60,-59.1], sub_lats=[-34.35,-33.7],
                  lons_points=-666, lats_points=-666)

#Extra
PlotTimeEvolution('T2',sub_dominio=True,
                  sub_lons=[-60,-59.1], sub_lats=[-34.35,-33.7],
                  lons_points=-666, lats_points=-666)
#----------------------------------------------------------------------------------------------------------------------#
#Ver dominio de la cuenca del rio salado
