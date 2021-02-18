'''
| This script generates a .geojson file with a MultiLineString
| geometry with a line for every route in the CARRIS
| network. From the GTFS specification we only get straight lines
| connecting the stops that exist in each particular route.
'''

import pandas as pd
import json

df = pd.read_csv('data/carris_gtfs/shapes.txt', sep=',', decimal='.')

lines = []

for shape in df['shape_id'].unique():
	sequence = df[ df['shape_id']==shape ]
	sequence.sort_values('shape_pt_sequence')
	
	line = []

	for _, row in sequence.iterrows():
		line.append([row['shape_pt_lon'], row['shape_pt_lat']])

	lines.append(line)

json_data = {
	'type': 'MultiLineString',
	'coordinates': lines
}

with open('data/lisbon_line_shapes_gtfs.geojson', 'w') as json_file:
	json.dump(json_data, json_file, indent=4)