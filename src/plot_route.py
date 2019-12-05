#!/usr/bin/env python3

import sys
import folium
import os
import pandas as pd

from main import parse_stop_indices 

STOPS_PATH = os.path.join('..', 'data', 'route_25_valid.csv')
LOCS_PATH = os.path.join('..', 'data', 'LongLat_of_AllStops.csv')
ID_COL = 'UniqueNum'


def load_stops(stops_path=STOPS_PATH):
    # parse the stop data
    stop_data = pd.read_csv(stops_path,
                            usecols=['Transfer', 'CorrespondingStopID', ID_COL])
    return stop_data

def load_locs(locs_path=LOCS_PATH):
    #parse the location data
    loc_data = pd.read_csv(locs_path,
                           usecols=['Lat', 'Long', ID_COL])
    return loc_data

def plot_route(stop_indices=None, indices_parsed=True, stop_data=None, loc_data=None):
    if stop_data == None:
        stop_data = load_stops()

    if loc_data == None:
        loc_data = load_locs()
    
    stop_locs = pd.merge(stop_data, loc_data, on=ID_COL)

    # Focus the map on the middle of our data set
    ave_lat = sum(loc_data['Lat']) / loc_data.shape[0]
    ave_long = sum(loc_data['Long']) / loc_data.shape[0]
    route_map = folium.Map(location=[ave_lat, ave_long], tiles='Stamen Toner', zoom_start=12)

    # Plot full bus route
    folium.PolyLine(loc_data[['Lat', 'Long']]).add_to(route_map)

    if stop_indices == None:
        # partition the stop data into current and candidate stops
        mask = stop_locs['CorrespondingStopID'].isna()
        candi_stops = stop_locs[mask]
        cur_stops = stop_locs[~mask]

        for i, row in cur_stops.iterrows():
            folium.Marker(
                location=row[['Lat', 'Long']],
                icon=folium.Icon(color='blue')
            ).add_to(route_map)
        
        for i, row in candi_stops.iterrows():
            folium.Marker(
                location=row[['Lat', 'Long']],
                icon=folium.Icon(color='red')
            ).add_to(route_map)

        route_map.save('all_stops_map.html')
    
    else:
        mask = stop_locs['Transfer'] == 'Yes'
        transfer = stop_locs[mask]
        non_transfer = stop_locs[~mask]

        if not indices_parsed:
            stop_indices = parse_stop_indices(stop_indices, non_transfer, transfer)

        for i, row in stop_locs.iloc[stop_indices].iterrows():
            folium.Marker(
                location=row[['Lat', 'Long']],
                icon=folium.Icon(color='blue')
            ).add_to(route_map)


def main():
    indices = None
    if len(sys.argv) == 2:
        indices = list(sys.argv[1])

    plot_route(indices, indices_parsed=False)

if __name__ == '__main__':
    main()
