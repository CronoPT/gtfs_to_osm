'''
| This script generates a .geojson file with a MultiPoint
| geometry with a point for every stop in the CARRIS
| network.
'''

import pandas as pd
import json

df = pd.read_csv('data/carris_gtfs/stops.txt', sep=',', decimal='.')

points = []

for _, row in df.iterrows():
    points.append([row['stop_lon'], row['stop_lat']])

json_data = {
    'type': 'MultiPoint',
    'coordinates': points
}

with open('data/lisbon_stops_gtfs.geojson', 'w') as json_file:
	json.dump(json_data, json_file, indent=4)