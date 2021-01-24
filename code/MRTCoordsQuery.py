###
# MRTCoordsQuery.py v0.0.1 2021-01-15
# --
#
# This file queries the LAT/LON for MRT station coordinates, using as its base
# file a "train-station-chinese-names.csv" file that is available from
# data.gov.sg.
###


import pandas as pd
from requests_futures.sessions import FuturesSession


# define paths and API Key to be used
root = '/Users/kwokhao/GoogleDrive/Research/mrt/sentient/'
git = '/Users/kwokhao/GoogleDrive/Research/mrt/hdb-amenities/'
APIKeyGoogle = 'REDACTED'  # sentient.io API Key (ends in QU)

# read csv file of train station names
dMRT = pd.read_csv(git + "data/train-station-chinese-names.csv")
lineKeys = ['NS', 'EW', 'NE', 'CC', 'DT']


def getLatLon(loc):
    '''gets coordinates of location query (string) using onemap.sg API. Currently
    not in use, but should replace the Google Maps query if sentient.io Google
    Maps API remains disabled.
    '''

    with FuturesSession(max_workers=16) as s:
        params = {'searchVal': loc,
                  'returnGeom': "Y",
                  "getAddrDetails": "Y"}
        r = s.get("https://developers.onemap.sg/commonapi/search",
                  params=params)
    coords = r.result().json()['results'][0]
    return coords['LATITUDE'], coords['LONGITUDE']


def getLineDict(stationCodes, lineKeys=lineKeys):
    'generate indices of dMRT that correspond to each MRT line'
    lineDict = {}
    for key in lineKeys:
        lineDict[key] = [idx for idx in range(len(stationCodes))
                         if stationCodes[idx][0:2] == key]
    return lineDict


def getStationQueryList(lineKey, lineDict, d=dMRT):
    '''obtains a list of MRT stations corresponding to the line `linekey`,
    of typical form `Jurong East MRT Station`
    '''
    stationNames = d.iloc[lineDict[lineKey]].mrt_station_english
    fullStationNames = [entry + " MRT Station" for entry in stationNames]
    return fullStationNames


# query station lat/lon
def getStationCoordinates(stationQueryList, APIKey=APIKeyGoogle):
    'gets GPS coordinates for the list of stations in `stationQueryList`'
    stationCoordsDict = {}
    for entry in stationQueryList:
        session = FuturesSession()
        query = 'https://maps.googleapis.com/maps/api/geocode/json?address=' \
                '{},+Singapore,+Singapore&key=' \
                '{}'.format(entry, APIKey)
        response = session.get(query).result()
        assert response.status_code == 200, "{}".format(response.reason)
        stationCoordsDict[entry] = list(
            response.json()['results'][0]['geometry']['location'].values())
    return stationCoordsDict


def genStationCSV(lineKey, lineDict, d=dMRT, APIKey=APIKeyGoogle):
    'outputs CSV corresponding to MRT line in `lineKey`'
    stationQueryList = getStationQueryList(lineKey, lineDict, d)
    stationCoordsDict = getStationCoordinates(stationQueryList, APIKey)
    dOutput = pd.DataFrame(
        list(stationCoordsDict.values()), columns=['LAT', 'LON'])
    # recover station names
    dOutput['station'] = [str(key)[:-12]
                          for key in list(stationCoordsDict.keys())]
    # recover station ID
    dOutput['id'] = [d[(d.mrt_station_english == name) &
                       (d.stn_code.str.contains(lineKey))].values[0][0]
                     for name in dOutput.station]
    dOutput.to_csv(git + 'data/{}LStationCoords.csv'.format(lineKey))


# run once
# lineDict = getLineDict(dMRT.stn_code)
# for key in lineDict.keys():
    # genStationCSV(key, lineDict)

# note: LAT/LON for Bukit Gombak MRT (NSL) and Bukit Panjang MRT (DTL)
# are wrong. I have manually corrected them in their respective CSVs

###
# stack station CSVs together
###

dMRT = pd.DataFrame()

for key in lineKeys:
    dMRT = dMRT.append(
        pd.read_csv(git + 'data/{}LStationCoords.csv'.format(key)
                    ).iloc[:, 1:5])

# mark interchanges
dMRT["interchange"] = dMRT.duplicated(["station"], keep=False)
dMRT["interchange"] = np.where(
    dMRT.id == "DT1", True, dMRT.interchange)  # correct for Bukit Panjang
dMRT.drop_duplicates(["station"], inplace=True)

dMRT.to_csv(git + 'data/MRTStationCoords.csv')
