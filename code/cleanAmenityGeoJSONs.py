###
# cleanAmenityGeoJSONs.py v0.1.1 JAN-15-2021
# ---
#
# This file cleans amenity geojsons, including supermarkets, hawkers,
# gyms, clinics and hotels.
###


###
# 1. IMPORT PACKAGES AND PATHS
###

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shapely
import geopandas as gpd
from geopandas.tools import sjoin
from shapely.geometry import Point, Polygon
import contextily as ctx
import re

# define default paths
root = '/Users/kwokhao/GoogleDrive/Research/mrt/sentient/'
git = '/Users/kwokhao/GoogleDrive/Research/mrt/hdb-amenities/'
dropbox = '/Users/kwokhao/Dropbox (Personal)/hdb-amenities/'

# TODO: add comments as to how frequently data are updated... and provenance
supermarketPath = git + "data/supermarkets/supermarkets-geojson.geojson"  # supermarkets, data.gov.sg
hawkerPath = git + "data/hawker-centres/hawker-centres-geojson.geojson"  # hawker centres, data.gov.sg (yearly?)
gymPath = git + "data/gymssg/gyms-sg-geojson.geojson"  # gyms, data.gov.sg (yearly? but not updated since 2019)
clinicPath = git + "data/chas-clinics/chas-clinics-geojson.geojson"  # clinics, data.gov.sg (yearly?)
hotelPath = git + "data/hotels/hotels-geojson.geojson"  # hotels, data.gov.sg (yearly?)
parkPath = git + "data/parks/parks-geojson.geojson"  # parks, data.gov.sg (yearly?)
ssoPath = git + "data/social-service-offices/social-service-offices-geojson.geojson"
ccPath = git + "data/community-clubs/community-clubs-geojson.geojson"

planningAreaPath = git + '/data/planning-area-census2010-shp/' + \
    'Planning_Area_Census2010.shp'
subzonePath = git + '/data/master-plan-2014-subzone-boundary-no-sea/' + \
    'master-plan-2014-subzone-boundary-no-sea-shp/' + \
    'MP14_SUBZONE_NO_SEA_PL.shp'

sg = gpd.read_file(planningAreaPath).to_crs(4326)
sgA = gpd.read_file(subzonePath).to_crs(4326).loc[
    :, ['PLN_AREA_N', 'SUBZONE_N', 'geometry']]


####
# 2. DATA CLEANING
# ---
#
# For each amenity type, data are imported directly from a GeoJSON file.
# Broadly, we take the following steps:
#
# 1. Strip HTML tag from "Description" field
# 2. Observe and extract fields from "Description" tag
# 3. Extract coordinates from geometry
# 4. Add other interesting characteristics (NTUC supermarket, chain supermarket
# etc)
# 5. Merge amenity data with subzone/planning area data
# 6. Export amenity data to .csv file
####

###
# Process supermarket data
###

# strip HTML tags from supermarket data
dSuper = gpd.read_file(supermarketPath)
dSuper.Description = dSuper.Description.str.replace(r"<[^>]*>", "")

# partition supermarket description by field

superFieldNames = ["LIC_NAME", "BLK_HOUSE", "STR_NAME", "UNIT_NO", "POSTCODE",
                   "LIC_NO", "INC_CRC", "FMEL_UPD_D"]
for i in range(len(superFieldNames)-1):
    dSuper[superFieldNames[i]] = dSuper.Description.str.partition(
        superFieldNames[i])[2].str.partition(superFieldNames[i+1])[0]

dSuper.drop("Description", axis=1, inplace=True)

# clean whitespace
for colName in dSuper.columns[2:]:
    dSuper[colName] = dSuper[colName].map(lambda row: row.strip())
dSuper["address"] = dSuper["BLK_HOUSE"] + " " + dSuper["STR_NAME"]

# label if NTUC (public supermarket)
dSuper["NTUC"] = ["NTUC" in name for name in dSuper.LIC_NAME]
dSuper["chain"] = [("SHENG SIONG" in name) or ("COLD STORAGE" in name) or
                   ("HAO MART" in name) or ("PRIME SUPERMARKET" in name) or
                   ("NTUC" in name) for name in dSuper.LIC_NAME]

# merge supermarkets with relevant subzone (contains planning area)
dSuperM = sjoin(dSuper, sgA, how="left")

# add coordinates
dSuperM["LON"] = dSuperM.geometry.x
dSuperM["LAT"] = dSuperM.geometry.y

# export to file
dSuperM.drop(['Name', 'index_right', 'INC_CRC_left', 'INC_CRC_right',
              'X_ADDR', 'Y_ADDR', 'SHAPE_Leng', 'SHAPE_Area',
              'PLN_AREA_C', 'SUBZONE_C', 'REGION_N', 'REGION_C'],
             axis=1, errors='ignore').to_csv(
    git + "data/supermarkets/supermarketsCleaned.csv", index=False)


###
# Process hawker data
###

# strip HTML tags from hawker data
dHawker = gpd.read_file(hawkerPath)
dHawker.Description = dHawker.Description.str.replace(r"<[^>]*>", "")
dHawker.Description = dHawker.Description.str.replace(r"HYPERLINK", "")
dHawker.Description = dHawker.Description.str.replace(r"^\s(.*)\s*$", "")

# partition hawker description by field
clinicFieldNames = ["ADDRESSBLOCKHOUSENUMBER", "LATITUDE", "EST_ORIGINAL_COMPLETION_DATE",
                    "STATUS", "CLEANINGSTARTDATE", "ADDRESSUNITNUMBER", "ADDRESSFLOORNUMBER",
                    "NO_OF_FOOD_STALLS", "REGION", "APPROXIMATE_GFA", "LONGITUDE",
                    "INFO_ON_CO_LOCATORS", "NO_OF_MARKET_STALLS", "AWARDED_DATE",
                    "LANDYADDRESSPOINT", "CLEANINGENDDATE", "PHOTOURL", "DESCRIPTION",
                    "NAME", "ADDRESSTYPE", "RNR_STATUS", "ADDRESSBUILDINGNAME",
                    "HUP_COMPLETION_DATE", "LANDXADDRESSPOINT", "ADDRESSSTREETNAME",
                    "ADDRESSPOSTALCODE", "DESCRIPTION_MYENV", "IMPLEMENTATION_DATE",
                    "ADDRESS_MYENV", "INC_CRC", "FMEL_UPD_D"]
for i in range(len(clinicFieldNames)-1):
    dHawker[clinicFieldNames[i]] = dHawker.Description.str.partition(
        clinicFieldNames[i])[2].str.partition(clinicFieldNames[i+1])[0]


# drop empty columns
dHawker.drop(columns=["Description", "ADDRESSUNITNUMBER", "ADDRESSFLOORNUMBER", "INFO_ON_CO_LOCATORS",
                      "AWARDED_DATE", "ADDRESSBUILDINGNAME", "IMPLEMENTATION_DATE",
                      "LANDYADDRESSPOINT", "LANDXADDRESSPOINT"],
             inplace=True)

# clean whitespace
for colName in dHawker.columns[3:]:
    dHawker[colName] = dHawker[colName].map(lambda row: row.strip())
dHawker.NO_OF_FOOD_STALLS = dHawker.NO_OF_FOOD_STALLS.astype(int)
dHawker[["LONGITUDE", "LATITUDE"]] = dHawker[["LONGITUDE", "LATITUDE"]].astype(float)

dHawker = dHawker.replace(["", "#N/A", "TBC"], np.nan)

dHawker[["CLEANINGSTARTDATE", "CLEANINGENDDATE", "HUP_COMPLETION_DATE"]] = dHawker[
    ["CLEANINGSTARTDATE", "CLEANINGENDDATE", "HUP_COMPLETION_DATE"]].applymap(
        lambda entry: pd.to_datetime(entry, format="%d/%m/%Y")
)

# clean opening date of hawker center
hawkerYearFlag = dHawker.EST_ORIGINAL_COMPLETION_DATE.map(
    lambda entry: re.match(r"^(19|20)[0-9]{2}$", str(entry)) is not None)

dHawker.EST_ORIGINAL_COMPLETION_DATE[
    hawkerYearFlag] = dHawker.EST_ORIGINAL_COMPLETION_DATE[
    hawkerYearFlag].map(lambda entry: "1/1/" + str(entry))


# match hawker centers with planning areas
dHawkerM = sjoin(dHawker.iloc[:, 0:-3], sg, how="left")

# add coordinates
dHawkerM["LON"] = dHawkerM.geometry.x
dHawkerM["LAT"] = dHawkerM.geometry.y

# export to file, but drop datetimes
dHawkerM.iloc[:, 2:].drop(
    columns=["CLEANINGSTARTDATE", "CLEANINGENDDATE", "HUP_COMPLETION_DATE",
             "X_ADDR", "Y_ADDR", "SHAPE_Leng", "SHAPE_Area",
             "index_right", "INC_CRC", "FMEL_UPD_D", "OBJECTID",
             "ADDRESSTYPE", "CA_IND", "PHOTOURL"]).to_csv(
                 "./data/hawker-centres/hawkersCleaned.csv")


###
# Process gym data
###

dGym = gpd.read_file(gymPath)
dGym.Description = dGym.Description.str.replace(r"<[^>]*>", "")
dGym.Description = dGym.Description.str.replace(r"HYPERLINK", "")
dGym.Description = dGym.Description.str.replace(r"^\s(.*)\s*$", "")

# A REMPLIR

###
# Process CHAS clinic data
###

dClinic = gpd.read_file(clinicPath, epsg=4326)
dClinic.Description = dClinic.Description.str.replace(r"<[^>]*>", "")
# dClinic.Description = dClinic.Description.str.replace(r"", "")
clinicFieldNames = ["HCI_CODE", "HCI_NAME", "LICENCE_TYPE", "HCI_TEL",
                    "POSTAL_CD", "ADDR_TYPE", "BLK_HSE_NO", "FLOOR_NO",
                    "UNIT_NO", "STREET_NAME", "BUILDING_NAME",
                    "CLINIC_PROGRAMME_CODE", "X_COORDINATE", "Y_COORDINATE",
                    "INC_CRC", "FMEL_UPD_D"]

for i in range(len(clinicFieldNames)-1):
    dClinic[clinicFieldNames[i]] = dClinic.Description.str.partition(
        clinicFieldNames[i])[2].str.partition(clinicFieldNames[i+1])[0]

dClinic['LON'] = dClinic.geometry.x
dClinic['LAT'] = dClinic.geometry.y
dClinic['geometry'] = [shapely.wkb.loads(
    shapely.wkb.dumps(point, output_dimension=2)) for point in dClinic.geometry]

dHotelM = sjoin(
    dClinic, sgA.loc[:, ['geometry', 'SUBZONE_N', 'PLN_AREA_N']], how="left")
dHotelM.drop(['Name', 'Description', 'index_right'], axis=1).to_csv(
    git + "data/chas-clinics/clinicsCleaned.csv", index=False)

# plot clinics
fig, ax = plt.subplots(figsize=(15, 9))
sg.plot(ax=ax, alpha=0.5, edgecolor="white")
ctx.add_basemap(ax, crs=4326)
dClinic.plot(ax=ax, alpha=0.5, color="green")

# A REMPLIR

###
# PROCESS HOTEL DATA
###

dHotel = gpd.read_file(hotelPath, epsg=4326)
dHotel.Description = dHotel.Description.str.replace(r"<[^>]*>", "")

hotelFieldNames = ["HYPERLINK", "DESCRIPTION", "POSTALCODE", "KEEPERNAME",
                   "TOTALROOMS", "ADDRESS", "INC_CRC", "FMEL_UPD_D",
                   "NAME"]

for i in range(len(hotelFieldNames)-1):
    dHotel[hotelFieldNames[i]] = dHotel.Description.str.partition(
        hotelFieldNames[i])[2].str.partition(hotelFieldNames[i+1])[0]

dHotel['LON'] = dHotel.geometry.x
dHotel['LAT'] = dHotel.geometry.y
dHotel['geometry'] = [shapely.wkb.loads(
    shapely.wkb.dumps(point, output_dimension=2)) for point in dHotel.geometry]
dHotelM = sjoin(
    dHotel, sgA.loc[:, ['geometry', 'SUBZONE_N', 'PLN_AREA_N']], how="left")
dHotelM.drop(['Name', 'Description', 'index_right'], axis=1).to_csv(
    git + "data/hotels/hotelsCleaned.csv", index=False)

###
# PROCESS PARK, SOCIAL SERVICE OFFICE AND COMMUNITY CLUB DATA
###


def fastExportBySubzone(
        path, fieldname, filename,
        exportCleaned=False, fieldNames=["LANDYADDRESSPOINT", " NAME ", "PHOTOURL"]):
    "Quickly exports a given .geojson file to a .csv file of quantities by subzone"

    d = gpd.read_file(path, crs='epsg:4326')
    d.Description = d.Description.str.replace(r"<[^>]*>", "")
    d['LON'] = d.geometry.x
    d['LAT'] = d.geometry.y
    d['geometry'] = [shapely.wkb.loads(
        shapely.wkb.dumps(point, output_dimension=2)) for point in d.geometry]

    dm = sjoin(d, sgA, how='left', op='within')
    if exportCleaned:
        for i in range(len(fieldNames)-1):
            dm[fieldNames[i].replace(" ", "")] = dm.Description.str.partition(
                fieldNames[i])[2].str.partition(fieldNames[i+1])[0].str.strip()
        dm.loc[:, ['NAME', 'LON', 'LAT', 'PLN_AREA_N', 'SUBZONE_N']].to_csv(
            dropbox + filename, index=False)
    else:
        dm = dm.groupby('SUBZONE_N').Name.count().reset_index()
        dm = dm.merge(sgA[['PLN_AREA_N', 'SUBZONE_N']], on='SUBZONE_N', how='left')
        dm.rename({'Name': fieldname}, axis=1).to_csv(dropbox + filename, index=False)


fastExportBySubzone(parkPath, 'parks', 'parksBySubzone.csv')
fastExportBySubzone(parkPath, 'parks', 'parksCleaned.csv', exportCleaned=True)
fastExportBySubzone(ssoPath, 'sso', 'ssoBySubzone.csv')
fastExportBySubzone(ccPath, 'cc', 'ccBySubzone.csv')


####
# PLOTTING
# ---
####

# plot supermarkets and hawker centres

plt.close()

fig, ax = plt.subplots(figsize=(15, 9))
sg.plot(ax=ax, alpha=0.5, edgecolor="white")
ctx.add_basemap(ax, crs=4326)

dSuper[dSuper.NTUC].plot(ax=ax, alpha=0.5, color="blue")
dSuper[~dSuper.NTUC].plot(ax=ax, alpha=0.5, color="orange")
dHawker.plot(ax=ax, alpha=0.5, color="green")
ax.legend(["NTUC Supermarket", "Other Supermarket", "Hawker Center"])


plt.show()
plt.savefig("./make_data/mapHawkerSuper.png", dpi=300, bbox_inches=0)
