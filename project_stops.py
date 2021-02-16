'''
| This script generates will project the bus stops 
| present in the GTFS specification to actual points
| on the road network using the algorithm described
| by Vuurstaek et al. 
'''

import pandas as pd
import json

stops_df = pd.read_csv('data/carris_gtfs/stops.txt', sep=',', decimal='.')
route_df = pd.read_csv('data/carris_gtfs/shapes.txt', sep=',', decimal='.')

stop_ids = []
not_stop = []
for index, row in route_df.iterrows():
    lat = row['shape_pt_lat']
    lon = row['shape_pt_lon']
    res = stops_df.loc[(stops_df['stop_lat']==lat) & (stops_df['stop_lon']==lon)]
    if res.shape[0] > 0:
        stop_ids.append(res.iloc[0]['stop_id'])
    else:
        stop_ids.append('Not Stop')
        not_stop.append([lon, lat])

route_df['route_id'] = stop_ids
route_df = route_df[route_df['route_id']!='Not Stop']

json_data = {
    'type': 'MultiPoint',
    'coordinates': not_stop
}

with open('data/not_stops_gtfs.geojson', 'w') as json_file:
	json.dump(json_data, json_file, indent=4)