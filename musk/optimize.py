import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pulp import *
from scipy.spatial import distance
import math
import geopandas as gpd
import pyproj
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader

desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 12)

# Variables
location = "NorthNorway"
traffic_data = location + "/tf_cs.csv"
park_data = location + "/park.csv"
road_shp = location + "/Roads/roads10.shp"
var_long = "Long"
var_lat = "Lat"
var_x = "x"
var_y = "y"

# Output of location shapefile
output = location

# Import GIS data and car park location data
GIS_data = pd.read_csv(traffic_data, sep=";", decimal=",").fillna(0)        # Contains Area of cell, traffic flow, position of centroids (x,y) in decimal degrees, and count of charging stations in grid (Count)
car_park_data = pd.read_csv(park_data, sep=";", decimal=",").fillna(0)      # Contains longitude and latitude position of parking lots (x, y).

GIS_df = pd.DataFrame(GIS_data)
car_park_df = pd.DataFrame(car_park_data)

# Cleaning datasets
GIS_df = GIS_df[["Shape_Area", "telling", "x", "y", "Count"]]
car_park_df = car_park_df[[var_long, var_lat]]

def gen_sets(df_demand, df_parking):
    """Generate sets to use in the optimization problem"""
    # set of charging demand locations (destinations)
    demand_lc = df_demand.index.tolist()
    # set of candidates for charging station locations (currently existing parking lots)
    chg_lc = df_parking.index.tolist()
    
    return demand_lc, chg_lc        # Returns a list of index numbers for df_demand and df_parking [0,...,x] and [0,...,y]

def gen_parameters(df_demand, df_parking):
    """Generate parameters to use in the optimization problem,
    including cost to install charging stations, operating costs and others..."""

    v0 = 0.178 #(21657+599169+455)/3497790 El-lorries,cars and vans divided by car fleet 0.016#
    u = 0.078   # the EV penetration rate (utilisation rate) 
    pe = 1.1   #0.3924   # price of electricity per kWh (kr/kWh)

    lj = 4  # Upper bound for chargers in station j
    alpha = 468  # Referance battery capacity (kWh) | 468 kWh for BET | 70 kWh for EV
    #N = num_of_CS      # Total number of stations to be installed

    Ai = df_demand["Shape_Area"]  # Ai stands for sum of area of the mixed use parts in cell i
    A = df_demand["Shape_Area"]              # A is the total area of cell i
    vi = Ai / A * v0                           # Where vi is the charging possibility of an EV in cell i
    fi = df_demand["telling"]          # Where fi is the average traffic flow in grid i
    di = u * vi * fi                           # Where di represents the charging demand of EV in grid i
    di = di.to_dict()
    ev_time = 1.8                  # Charging time ev in hours

    # Rapid Chargers
    df_demand['m'] = 7                       # Number of charging sessions per day (session/day)
    m = df_demand['m'].to_dict()
    df_demand['p'] = (7.69*alpha)/ev_time                   # Cost of charging price per hour NOK/minute (cancels out in the objective function)
    p = df_demand['p'].to_dict()
    df_demand['t'] = ev_time                    # Charging time for an EV (minutes)
    t = df_demand['t'].to_dict()
    df_demand['ci_j'] = 1437500/365                 #Total Ivestment cost NOK
    ci_j = df_demand['ci_j'].to_dict()
    df_demand['cr_j'] = 400                 # cr_j represents the parking fee per day of parking lot j NOK
    cr_j = df_demand['cr_j'].to_dict()
    #df_demand['ce_j'] = 1437500/2/365                # ce_j represents the price of a charger in station j NOK
    #ce_j = df_demand['ce_j'].to_dict()
    #
    # distance matrix of charging station location candidates and charging demand location
    #coords_parking
    c1 = np.array([(x, y) for x, y in zip(df_parking[var_long], df_parking[var_lat])])

    #coords_demand
    c2= np.array([(x, y) for x, y in zip(df_demand[var_x], df_demand[var_y])])

    # create projections, using a mean (lat, lon) for aeqd
    lat_0, lon_0 = np.mean(np.append(c1[:,0], c2[:,0])), np.mean(np.append(c1[:,1], c2[:,1]))
    proj = pyproj.Proj(proj='aeqd', lat_0=lat_0, lon_0=lon_0, x_0=lon_0, y_0=lat_0)
    WGS84 = pyproj.Proj(init='epsg:4326')

    # transform coordinates
    projected_c1 = pyproj.transform(WGS84, proj, c1[:,1], c1[:,0])
    projected_c2 = pyproj.transform(WGS84, proj, c2[:,1], c2[:,0])
    projected_c1 = np.column_stack(projected_c1)
    projected_c2 = np.column_stack(projected_c2)

    # calculate pairwise distances in km with both methods
    distance_matrix2 = distance.cdist(projected_c1, projected_c2)
    distance_matrix3 = distance_matrix3 = pd.DataFrame(distance_matrix2, index=df_parking.index.tolist(), columns=df_demand.index.tolist())
    #distance_matrix3.to_csv("NorthNorway/distance_matrix.csv")
    #print(distance_matrix3)
    ce_j = 0
    return di, m, p, t, ci_j, cr_j, ce_j, pe, alpha, lj, distance_matrix3
    
def gen_demand(df_demand):
    """generate the current demand for charging for each cell i"""
    #df_demand['zero_cs'] = 0                 # ce_j represents the price of a charger in station j
    #diz = df_demand['zero_cs'].to_dict()
    #print(diz)
    diz = df_demand["Count"]  # Number of existing chargers in cell i
    diz = diz.to_dict()
    return diz

def plot_map():
    fname = 'Norway/no/gadm41_NOR_0.shp'
    adm1_shapes = list(shpreader.Reader(fname).geometries())
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(resolution='50m')
    ax.add_geometries(adm1_shapes, ccrs.PlateCarree(),
                  edgecolor='#c9c9c9', facecolor='#c9c9c9', alpha=0.5)
    ax.set_extent([4, 35, 47, 80], ccrs.PlateCarree())

def plot_CS(opt_loc_df, opt_loc_df2, N):
    """ Import the road shapefiles and plot map """
    shp_path_roads_1 = gpd.read_file(road_shp)
    shp_path_roads_1 = shp_path_roads_1.to_crs(epsg=4326)

    base = shp_path_roads_1.plot(figsize=(12, 8), color='grey', lw=0.4, zorder=0)
    #base = plot_map()
    plot = sns.scatterplot(ax=base, x=opt_loc_df['Long'], y=opt_loc_df['Lat'], color='dodgerblue', legend='full')

    plot.set_title(f'Optimal locations for {int(N.varValue)} chargers in ' + location + "- no existing CS")
    
    for line in range(0, opt_loc_df2.shape[0]):
        plot.text(opt_loc_df2.Long[line] + 50, opt_loc_df2.Lat[line],
                  opt_loc_df2.value[line], horizontalalignment='left',
                  size='medium', color='black', weight='semibold')
    plt.legend(["Road network", "New charging stations"])
    plt.show()


    gdf = gpd.GeoDataFrame(opt_loc_df, geometry=gpd.points_from_xy(x=opt_loc_df.Long,y=opt_loc_df.Lat))
    gdf.crs = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
    print(gdf)
    opt_loc_df.to_csv("location.csv")
    output_file = "shape/" + "opt_loc_" + output + "_CS_" + str(N) + ".shp"
    gdf.to_file(output_file, driver='ESRI Shapefile')

def optimize(df_demand, df_parking):
    """ Brings all the variables together and optimizes the problem """
    # Import i and j set function
    demand_lc, chg_lc = gen_sets(df_demand, df_parking)

    # Import parameters function
    di, m, p, t, ci_j, cr_j, ce_j, pe, alpha, lj, distance_matrix = gen_parameters(df_demand, df_parking)

    # Import current demand of car park z in cell i
    diz = gen_demand(df_demand)

     # set up the optimization problem
    prob = LpProblem('FacilityLocation', LpMaximize)

    n = LpVariable.dicts("no_of_chgrs_station_j",
                         [j for j in chg_lc],
                         0, lj, LpInteger)
    q = LpVariable.dicts("Remaining_dem_station_j",
                         [j for j in chg_lc],
                         0)
    c = LpVariable.dicts("Tot_costs_station_j",
                         [j for j in chg_lc],
                         0)
    #aid = LpVariable("Incentive")

    N = LpVariable("New_stations")
    #N = 10
    df_demand['insentive'] = 1437500/365*0.8                 # ce_j represents the price of a charger in station j
    ins_nj = df_demand['insentive'].to_dict()
    x = LpVariable.dicts("UseLocation", [j for j in chg_lc], 0, 1, LpBinary)

    r = np.full([len(demand_lc), len(chg_lc)], None)

    for i in demand_lc:
        for j in chg_lc:
            if distance_matrix[i][j] <= 500:
                r[i][j] = 1
            else:
                r[i][j] = 0
    count = np.count_nonzero(r == 1)
    print("The number of potential connections with a distance less than 500m is:", count)
    probloop = chg_lc
    incentive = 100*10**6
     
    # Objective function
    prob += lpSum(p[j]*t[j] * q[j] - c[j] for j in probloop)
    
    # Create empty dictionary for the remaining demand in cell i
    zip_iterator = zip(demand_lc, [None]*len(demand_lc))
    dr = dict(zip_iterator)

    # For each cell i subtract the existing number of charging stations from the charging demands in cell i
    for i in demand_lc:
        for j in chg_lc:
            dr[i] = di[i] - diz[i] * m[j]
            if dr[i] < 0:       # Can't have negative demand therefore limit minimum demand to zero
                dr[i] = 0
    #print(dr)
    
    # Constraints
    for j in probloop:
        prob += c[j] == (cr_j[j] + ci_j[j] - ins_nj[j]) * n[j] + q[j]*(pe * alpha)  # Calculation of cost
    for j in probloop:
        prob += q[j] <= n[j] * m[j]                                 # Constraint 1
    for j in probloop:
        prob += q[j] <= lpSum(r[i][j] * dr[i] for i in demand_lc)   # Constraint 2
    for i in probloop:
        prob += lpSum(x[j] * r[i][j] for j in chg_lc) <= 1          # Constraint 3
    for j in probloop:
        prob += n[j] >= x[j]                                        # Constraint 4
    for j in probloop:
        prob += n[j] <= lj * x[j]                                   # Constraint 5

    prob += lpSum(x[j] for j in probloop) == N                      # Constraint 6
                                                            
    prob += lpSum(ins_nj[j] * n[j] for j in probloop) <= incentive/365      # Constraint 7
    
    # Run the optimization
    prob.solve()
    
    # Print status
    print("Status: ", LpStatus[prob.status])
    tolerance = .00001
    opt_location = []
    for j in chg_lc:
        if x[j].varValue > tolerance:   # If binary value x is positive then the car park has been selected
            opt_location.append(j)
            #print("Establish charging station at parking lot", j)
    df_status = pd.DataFrame({"status": [LpStatus[prob.status]], "Tot_no_chargers": [len(opt_location)]})
    print("Final Optimisation Status:\n", df_status)
    
    # Add chargers to dict
    varDic = {}
    for variable in prob.variables():
        var = variable.name
        if var[:5] == 'no_of':      # Filter to obtain only the variable 'no_of_chgrs_station_j'
            varDic[var] = variable.varValue
    number_of_chargers = sum(varDic.values())
    

    var_df = pd.DataFrame.from_dict(varDic, orient='index', columns=['value'])
    # Sort the results numerically
    sorted_df = var_df.index.to_series().str.rsplit('_').str[-1].astype(int).sort_values()
    var_df = var_df.reindex(index=sorted_df.index)
    var_df.reset_index(inplace=True)

    location_df = pd.DataFrame(opt_location, columns=['opt_car_park_id'])
#     print(location_df.head())
#     print(car_park_df.head())
    opt_loc_df = pd.merge(location_df, car_park_df, left_on='opt_car_park_id',  right_index=True, how='left')
    opt_loc_df2 = pd.merge(opt_loc_df, var_df, left_on='opt_car_park_id',  right_index=True, how='left')
#     opt_loc_df2.to_csv(path_or_buf='optimal_locations.csv')

    print("n_j: ",number_of_chargers)
    plot_CS(opt_loc_df, opt_loc_df2, N)
    
    return opt_location, df_status

# Run the optimization function using the data set in GIS_df and car_park_df
optimize(GIS_df,car_park_df)

