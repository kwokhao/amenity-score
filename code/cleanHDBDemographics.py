###
# cleanHDBDemographics.py v0.0.1 25-AUG-2020
# --
#
# This file generates the demographic regressors needed for the
# amenity score NLLS. As a reminder, each amenity score weight behaves
# like `α_ia = α_a + α_a1 FracOld_i + α_a2 FracYoung_i`, where the old
# are 70+ and the young are 0-9.
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
dAdd = pd.read_csv(root + 'HDBAddress.csv').drop(['XCOORD', 'YCOORD'],
                                                 errors='ignore')
gdAdd = gpd.GeoDataFrame(
    dAdd, geometry=gpd.points_from_xy(dAdd.LON, dAdd.LAT), crs="epsg:4326"
)

# read demographics data. columns: ['postal_code', 'gender',
# 'age_group', 'residents']
dD = pd.read_csv(git + 'data/Gender_AgeRange01.csv')
dD["tot"] = dD.groupby("postal_code").residents.transform(sum)
dD["frac"] = dD.residents/dD.tot
dD["old"] = np.where(
    dD.age_group.str.contains("70"), 1, np.nan) * dD.frac
dD["fracOld"] = dD.groupby("postal_code").old.transform(np.nansum)
dD["young"] = np.where(
    dD.age_group.str.contains("0 TO 9"), 1, np.nan) * dD.frac
dD["fracYoung"] = dD.groupby("postal_code").young.transform(np.nansum)
dD.drop(["old", "young"], axis=1, inplace=True, errors='ignore')

dD.to_csv(git + "make_data/cleanedHDBDemographics.csv", index=False)
