###
# Python plots for amenity score
###

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# IMPORT geopandas and dependencies
import geopandas as gpd
# from shapely.geometry import Point, Polygon
import contextily as ctx
# from pyproj import Transformer

import seaborn as sns
sns.set(style="whitegrid", palette="pastel", color_codes=True)
# sns.mpl.rc("figure", figsize=(15, 9))

###
# PLOTTING
###

# define default paths
root = '/Users/kwokhao/GoogleDrive/Research/mrt/sentient/'
git = '/Users/kwokhao/GoogleDrive/Research/mrt/hdb-amenities/'
DROPBOX_PATH = "/Users/kwokhao/Dropbox (Personal)/hdb-amenities/"
BTO_PATH = "/Users/kwokhao/Dropbox (Princeton)/bto_singapore/"

# read from dAdd csv
dAdd = pd.read_csv(
    git + '/make_data/HDBAddressAmenityScore2020-05-01.csv')
gdAdd = gpd.GeoDataFrame(
    dAdd, geometry=gpd.points_from_xy(dAdd.LON, dAdd.LAT), crs="epsg:4326")

# read downtown line stations
dDTL = pd.read_csv(git + '/data/DTLStationCoords.csv').iloc[:, 1:]
gdDTL = gpd.GeoDataFrame(
    dDTL, geometry=gpd.points_from_xy(dDTL.LON, dDTL.LAT), crs="epsg:4326")

# shape file for singapore planning subzones (2014) NOTE: the coordinates
# are in the 3414(SVY21) format, but lat/lon are in the 4326(WGS84)
# format.

sg = gpd.read_file(
    git + '/data/master-plan-2014-subzone-boundary-no-sea/' +
    'master-plan-2014-subzone-boundary-no-sea-shp/' +
    'MP14_SUBZONE_NO_SEA_PL.shp').to_crs(4326)


APIKey = "AIzaSyDuk0177fbNsCY5Jr1Z55hVIkE5qg5ZgQU"  # sentient.io API Key for Google Maps


# compute convenient variables
def normAS(colAS):
    return colAS/np.max(colAS)


gdAdd['normAmenityScore2'] = normAS(gdAdd.amenity_score)


# plot my home postal code with geopandas
def pointPlotter(sg=sg, gdf=gdAdd):
    fig, ax = plt.subplots(figsize=(15, 9))
    sg.plot(ax=ax, alpha=0.5, edgecolor="white")
    ctx.add_basemap(ax, crs=4326)
    sg.iloc[[104, 104]].plot(ax=ax, alpha=0.3, color="orange")
    gdf.iloc[[1734, 1734]].plot(ax=ax, alpha=0.5, color="red")


try:
    plt.close()
except:
    pass

pointPlotter()
plt.show()


# plot bottom and top quartile (50-50) amenity scores
def plotAmenityScore(amenityScoreName='normAmenityScore1', sg=sg, gdf=gdAdd):
    '''plots all stations, optionally highlighting DTL stage 2 stations.
    '''
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_axis_off()
    sg.plot(ax=ax, alpha=0.5, edgecolor="white")
    ctx.add_basemap(ax, crs=4326)
    gdAdd.plot(column=amenityScoreName, ax=ax, alpha=0.1, markersize=10, legend=True)
    # crop plot
    plt.subplots_adjust(top=0.98, bottom=0.02, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)


try:
    plt.close()
except:
    pass

plotAmenityScore()
plt.show()
plt.savefig(git + "/make_data/map50-50.png", dpi=300, bbox_inches=0)
plt.close()

plotAmenityScore('normAmenityScore2')
plt.show()
plt.savefig(git + "/make_data/map33-33-33.png", dpi=300, bbox_inches=0)
plt.close()

###
# Repeat linear regression exercise with housing prices
###


def cleanStreetNames(df):
    'cleans the street names in the HDB transaction data'
    streetNamesToClean = df['street_name']

    # clean cardinal directions FIRST
    streetNamesToClean = streetNamesToClean.str.replace(
        r'NTH', r'NORTH').str.replace(r'STH', r'SOUTH')
    # clean road names
    # first replace the avenues
    streetNamesToClean = streetNamesToClean.str.replace(
        r'\sAVE\s', r' AVENUE ').str.replace(r'\sAVE$', r' AVENUE').str.replace(
            r'\sCL', r' CLOSE').str.replace(r'\sCRES', r' CRESCENT').str.replace(
                r'\sCTRL', r' CENTRAL').str.replace(r'\sCTR', ' CENTRE').str.replace(
                    r'\sDR', r' DRIVE').str.replace(r'JLN\s', r'JALAN ').str.replace(
                    r'LOR\s', r'LORONG ').str.replace(r'\sPL$', r' PLACE')
    streetNamesToClean = streetNamesToClean.str.replace(
        r'\sPK', r' PARK').str.replace(r'\sRD', r' ROAD')
    streetNamesToClean = streetNamesToClean.str.replace(
        r'\sST', r' STREET').str.replace(r'\sTER', r' TERRACE')
    # clean location contractions
    streetNamesToClean = streetNamesToClean.str.replace(
        r'BT\s', r'BUKIT ').str.replace(r'C\'WEALTH', r'COMMONWEALTH').str.replace(
            r'UPP\s', r'UPPER ').str.replace(r'\sHTS', r' HEIGHTS').str.replace(
                r'^TG\s', 'TANJONG ').str.replace(r'\sGDNS', r' GARDENS').str.replace(
                    r'^KG\s', 'KAMPONG ').str.replace('\sMKT\s', ' MARKET ')

    # finally set cleaned street names
    df['street_name'] = streetNamesToClean
    return df


def getFlatsWithAmenities(dFlats):
    dFlats = cleanStreetNames(dFlats)
    # merge data
    dFlatsWithAmenities = dFlats.merge(
        dAdd, left_on=['block', 'street_name'], right_on=['BLOCK', 'STREET'],
        validate='m:1', indicator=True)
    print(dFlatsWithAmenities.columns)
    # clean data
    dF = dFlatsWithAmenities.drop(
        ['BLOCK', 'STREET', 'XCOORD', 'YCOORD', '_merge'], axis=1).copy()
    dF.flat_type = [6 if rooms[0] == 'E' else 7 if rooms[0] == 'M' else int(
        rooms[0]) for rooms in dF.flat_type]
    dF.month = pd.to_datetime(dF.month)
    # save data
    # dF.to_stata(git + 'data/FlatsWithAmenities2015-.dta')
    dF.iloc[:, 0:14].to_stata(BTO_PATH + 'ResaleFlatsMerged2000-.dta')


dFlats = pd.read_stata(DROPBOX_PATH + "resale-flat-prices/resaleFlatPricesJan2000ToEnd.dta")
# dFlats = pd.read_csv(DROPBOX_PATH +
#     'resale-flat-prices/resale-flat-prices-based-on-registration-date-from-jan-2015-onwards.csv')
# getFlatsWithAmenities(dFlats)
