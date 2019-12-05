#!/usr/bin/env python3

import sys
import folium
import os
import pandas as pd

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

def plot_route(stop_indices=None, indices_parsed=True, stop_data=None, loc_data=None, out_path=None):
    if stop_data is None:
        stop_data = load_stops()

    if loc_data is None:
        loc_data = load_locs()
    
    stop_locs = pd.merge(stop_data, loc_data, on=ID_COL)

    # Focus the map on the middle of our data set
    ave_lat = sum(loc_data['Lat']) / loc_data.shape[0]
    ave_long = sum(loc_data['Long']) / loc_data.shape[0]
    route_map = folium.Map(location=[ave_lat, ave_long], tiles='Stamen Toner', zoom_start=12)

    # Plot full bus route
    folium.PolyLine(loc_data[['Lat', 'Long']]).add_to(route_map)

    if stop_indices is None:
        # partition the stop data into current and candidate stops
        mask = stop_locs['CorrespondingStopID'].isna()
        candi_stops = stop_locs[mask]
        cur_stops = stop_locs[~mask]
        
        transfer = cur_stops[cur_stops['Transfer'] == 'Yes']
        non_transfer = cur_stops[cur_stops['Transfer'] == 'No']
        
        # Plot all transfer stops as blue icons
        for i, row in transfer.iterrows():
            folium.Marker(
                location=row[['Lat', 'Long']],
                icon=folium.Icon(color='green')
            ).add_to(route_map)

        # Plot all current, non-transfer stops as blue icons
        for i, row in non_transfer.iterrows():
            folium.Marker(
                location=row[['Lat', 'Long']],
                icon=folium.Icon(color='blue')
            ).add_to(route_map)
        
        # Plot all candidate stops as red icons
        for i, row in candi_stops.iterrows():
            folium.Marker(
                location=row[['Lat', 'Long']],
                icon=folium.Icon(color='red')
            ).add_to(route_map)

        if out_path is None:
            route_map.save(os.path.join('..', 'plots', 'all_stops_map.html'))
    
    else:
        transfer = stop_locs[stop_locs['Transfer'] == 'Yes']
        non_transfer = stop_locs[stop_locs['Transfer'] == 'No']

        if not indices_parsed:
            print(sum(stop_indices))
            from main import parse_stop_indices 
            stop_indices = parse_stop_indices(stop_indices, non_transfer, transfer)
            print(len(stop_indices))

        for i, row in stop_locs.iloc[stop_indices].iterrows():
            folium.Marker(
                location=row[['Lat', 'Long']],
                icon=folium.Icon(color='blue')
            ).add_to(route_map)

        if out_path is None:
            route_map.save(os.path.join('..', 'plots', 'solution_map.html'))
    
    if out_path is not None:
        route_map.save(out_path)

def parse_list(string):
    # Get rid of brackets on either end
    string = string.replace(']','').replace('[','')
    
    # Split string up by the commas
    string_list = string.split(', ')

    # convert entries into ints
    int_list = [int(n) for n in string_list]

    return int_list

def main(): 
    indices = None
    if len(sys.argv) == 2:
        indices = parse_list(sys.argv[1])

    plot_route(indices, indices_parsed=False)

if __name__ == '__main__':
    main()
