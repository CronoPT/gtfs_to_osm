'''
| This script implements the projStops step in Vuurstaek et al.
| It does so by first indexing every road segment to a square
| by dividing the space spanned by the network into a grid.
| This indexing allows us to compute projections to a small
| amount of road segments instead of them all which takes hours
| as oposed to this script which is much faster.
'''


import pandas as pd
import numpy as np
import math
import networkx as nx
import utils.general_utils
import utils.geometric_utils
import utils.json_utils
import configs


def coord_to_square(lon, lat):
	'''
	| This function maps a coordinate pair to 
	| a x&y position in the greed.
	| x indexes longitude and y indexes latitude.
	'''
	x = math.floor( (lon-min_lon) / lon_step )
	y = math.floor( (lat-min_lat) / lat_step )
	x = x if x < configs.DIVISIONS else x-1
	y = y if y < configs.DIVISIONS else y-1
	return x, y


def fill_structures(origin, destin, struct, or_id, de_id, key):
	'''
	| This function will sort every road segment into the squares
	| on the grid that it spans.
	'''

	# this assures that we will have the origin point always higher
	origin, destin = (origin, destin) if origin[1] > destin[1] else (destin, origin)

	or_x, or_y = coord_to_square(origin[0], origin[1])
	de_x, de_y = coord_to_square(destin[0], destin[1])

	# if the origin and destiny are in the same square, that's the only square affected
	if or_x==de_x and or_y==de_y:
		struct[or_x][or_y]['links'].append({
			'segment':   [origin, destin],
			'origin_id': or_id,
			'destin_id': de_id,
			'key': key
		})
		return

	curr_x, curr_y = or_x, or_y
	moved = False
	while curr_x!=de_x or curr_y!=de_y:
		'''
		| The idea of this cycle is to successivelly aproxymate
		| our 'current point' with our destiny, every square
		| we go through is marked as containing our road
		| segmet origin-destin.
		'''

		# the limits of the current square
		max_lat_curr = struct[curr_x][curr_y]['max_lat']
		min_lat_curr = struct[curr_x][curr_y]['min_lat']
		max_lon_curr = struct[curr_x][curr_y]['max_lon']
		min_lon_curr = struct[curr_x][curr_y]['min_lon']

		origi_dest = [origin, destin]
		right_edge = [[max_lon_curr, max_lat_curr], [max_lon_curr, min_lat_curr]]
		lower_edge = [[max_lon_curr, min_lat_curr], [min_lon_curr, min_lat_curr]]
		lefty_edge = [[min_lon_curr, min_lat_curr], [min_lon_curr, max_lat_curr]]

		'''
		| We don't need to check the upper edge since we know that
		| the origin point is always higher than the destiny point.
		'''

		if utils.geometric_utils.lines_intersect(origi_dest, right_edge) and not moved:
			if curr_x<de_x:
				curr_x += 1
				moved   = True

		if utils.geometric_utils.lines_intersect(origi_dest, lower_edge) and not moved:
			if curr_y>de_y:
				curr_y -= 1
				moved   = True

		if utils.geometric_utils.lines_intersect(origi_dest, lefty_edge) and not moved:
			if curr_x>de_x:
				curr_x -= 1
				moved   = True

		moved = False
		struct[curr_x][curr_y]['links'].append({
			'segment':   [origin, destin],
			'origin_id': or_id,
			'destin_id': de_id,
			'key': key
		})


def dead_end_edge(road_net, segment):
	edge_info = road_net.get_edge_data(
		segment['origin_id'],
		segment['destin_id'],
		segment['key']
	)

	no_way_out = road_net.out_degree(segment['destin_id'])==0
	no_way_in  = road_net.in_degree(segment['origin_id'])==0
	one_way    = edge_info['oneway']==True

	return (no_way_out or no_way_in) and one_way


if __name__ == '__main__':

	net = utils.json_utils.read_json_object(configs.TWEAKED_NETWORK)
	road_net = utils.json_utils.read_network_json(configs.TWEAKED_NETWORK)

	max_lat = -np.inf
	min_lat = np.inf
	max_lon = -np.inf
	min_lon = np.inf

	# Just assessing the limits of the grid
	for node in net['nodes']:
		lon = node['x']
		lat = node['y']

		max_lat = lat if lat > max_lat else max_lat
		min_lat = lat if lat < min_lat else min_lat
		max_lon = lon if lon > max_lon else max_lon
		min_lon = lon if lon < min_lon else min_lon
		
	bounds = [[{} for i in range(configs.DIVISIONS)] for i in range(configs.DIVISIONS)]
	lat_step = (max_lat-min_lat)/configs.DIVISIONS
	lon_step = (max_lon-min_lon)/configs.DIVISIONS

	stops_df = pd.read_csv('data/gtfs/carris/stops.txt', sep=',', decimal='.')
	route_df = pd.read_csv('data/gtfs/carris/shapes.txt', sep=',', decimal='.')
	outub_df = pd.read_csv('data/PercursosOutubro2019.csv', sep=';', decimal=',', low_memory=False)

	stops = []
	for stop in outub_df['cod_paragem'].unique():
		res = outub_df[ outub_df['cod_paragem']==stop ]
		row = res.iloc[0]
		stops.append({
			'stop_id': int(row['cod_paragem']),
			'lat': float(row['latitude']),
			'lon': float(row['longitude'])
		})

	for x in range(configs.DIVISIONS):
		for y in range(configs.DIVISIONS):
			bounds[x][y] = {
				'max_lat': min_lat + (y+1)*lat_step,
				'min_lat': min_lat + y*lat_step,
				'max_lon': min_lon + (x+1)*lon_step,
				'min_lon': min_lon + x*lon_step,
				'links':   []
			}

	for index, adjacencies in enumerate(net['adjacency']):
		utils.general_utils.print_progress_bar(
			index, 
			len(net['adjacency']), 
			prefix='[MAPPING] 1/2'
		)
		for link in adjacencies:
			'''
			| Here we are just iterating over every road segment and 
			| sorting it by the squares it spans. The function that 
			| does the main job is 'fill_structures'.
			'''

			destin_item = list(filter(lambda item: item['id']==link['id'], net['nodes']))[0]
			or_id = net['nodes'][index]['id']
			de_id = destin_item['id']
			key   = 0 if 'key' not in link else link['key'] 
			if 'geometry' in link:
				origin = None
				destin = None
				for coords in link['geometry']:
					destin = coords
					if origin != None:
						fill_structures(origin, destin, bounds, or_id, de_id, key)
					origin = destin
			else:
				origin = [net['nodes'][index]['x'], net['nodes'][index]['y']]
				destin = [destin_item['x'], destin_item['y']]
				fill_structures(origin, destin, bounds, or_id, de_id, key)

	utils.general_utils.print_progress_bar(
		len(net['adjacency']), 
		len(net['adjacency']), 
		prefix='[MAPPING] 1/2'
	)

	stop_mappings = []
	stop_points   = []
	for index, stop in enumerate(stops):
		'''
		| Now we well actually project the stations on the road.
		| The whole process of sorting the segments in squares is
		| to make this faster. Because we have the squares we can 
		| project the station just in the vacinity of the station
		| instead of the whole network.
		'''

		utils.general_utils.print_progress_bar(
			index, 
			len(stops), 
			prefix='[PROJECT] 2/2'
		)

		lat = stop['lat']
		lon = stop['lon']
		stp = [lon, lat]
		pjs = [] 

		stop_points.append([lon, lat])
		stp_x, stp_y = coord_to_square(lon, lat)
		candidate_coords = [[stp_x, stp_y]]
		
		if stp_x > 0:
			candidate_coords.append([stp_x-1, stp_y])

		if stp_y > 0:
			candidate_coords.append([stp_x, stp_y-1])

		if stp_x < configs.DIVISIONS-1:
			candidate_coords.append([stp_x+1, stp_y])
		
		if stp_y < configs.DIVISIONS-1:
			candidate_coords.append([stp_x, stp_y+1])

		if stp_x>0 and stp_y>0:
			candidate_coords.append([stp_x-1, stp_y-1])

		if stp_x<configs.DIVISIONS-1 and stp_y<configs.DIVISIONS-1:
			candidate_coords.append([stp_x+1, stp_y+1])

		if stp_x>0 and stp_y<configs.DIVISIONS-1:
			candidate_coords.append([stp_x-1, stp_y+1])

		if stp_x<configs.DIVISIONS-1 and stp_y>0:
			candidate_coords.append([stp_x+1, stp_y-1])

		added_thresh = 0
		while len(pjs) == 0:
			for candidate in candidate_coords:
				for segment in bounds[candidate[0]][candidate[1]]['links']:
					closest  = utils.geometric_utils.closest_point(
						segment['segment'][0], 
						segment['segment'][1], 
						stp
					)
					distance = utils.geometric_utils.haversine_distance(stp, closest)
					if distance < configs.THRESHOLD+added_thresh and not dead_end_edge(road_net, segment):
						pjs.append({
							'point': closest,
							'distance':  distance,
							'origin_id': segment['origin_id'],
							'destin_id': segment['destin_id'],
							'key': segment['key']
						})
			added_thresh += 15

		'''
		| Because of the indexing by square on the grid, some
		| projections will be added twice, this cycle handles
		| those situations.
		'''
		filtered_pjs = []
		points_in_filtered = []
		for pj in pjs:
			if pj['point'] not in points_in_filtered: 
				filtered_pjs.append(pj)
				points_in_filtered.append(pj['point'])

		segments_projections = {}
		for pj in filtered_pjs:
			key = (pj['origin_id'], pj['destin_id'], pj['key'])

			if key not in segments_projections:
				segments_projections[key] = []

			segments_projections[key].append(pj)

		final_pjs = []
		for key, pjs in segments_projections.items():
			best_distance = np.inf
			best_proj = None
			for pj in pjs:
				if pj['distance']<best_distance:
					best_proj = pj
					best_distance = pj['distance']
			final_pjs.append(best_proj)

		stop_mappings.append({
			'stop_id':  stop['stop_id'],
			'point':    stp,
			'mappings': final_pjs
		})

	utils.general_utils.print_progress_bar(
		len(stops), 
		len(stops), 
		prefix='[PROJECT] 2/2'
	)

	utils.json_utils.write_json_object(configs.CANDIDATE_MAPPINGS, stop_mappings)
	utils.json_utils.write_geojson_points(configs.ORIGINAL_STOP_LOCATIONS, stop_points)
