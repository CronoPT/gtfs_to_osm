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
import json
import math
import networkx as nx

def print_progress_bar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
	'''
	| Call in a loop to create terminal progress bar
	| @params:
	|    iteration   - Required  : current iteration (Int)
	|    total       - Required  : total iterations (Int)
	|    prefix      - Optional  : prefix string (Str)
	|    suffix      - Optional  : suffix string (Str)
	|    decimals    - Optional  : positive number of decimals in percent complete (Int)
	|    length      - Optional  : character length of bar (Int)
	|    fill        - Optional  : bar fill character (Str)
	|    printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
	'''

	percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
	filledLength = int(length * iteration // total)
	bar = fill * filledLength + '-' * (length - filledLength)
	print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
	if iteration == total: 
		print()


def closest_point(a, b, p):
	'''                                __
	| This function finds the point in ab 
	| that is closest to point p (outside
	| the line segment).
	'''

	a = np.array(a)
	b = np.array(b)
	p = np.array(p)
	t = - np.inner((b-a), (a-p)) / np.inner((b-a), (b-a))

	if t > 0 and t < 1:
		return ((1-t)*a + t*b).tolist() 

	g0 = np.linalg.norm((a - p))
	g1 = np.linalg.norm((b - p))

	return a.tolist() if g0 < g1 else b.tolist()


def haversine_distance(a, b):
	'''
	| This function computes the haversine distance
	| between point a and point b. The distance is
	| in kilometers.
	'''

	lat1 = a[1]
	lat2 = b[1]
	lon1 = a[0]
	lon2 = b[0]
	lat1, lat2, lon1, lon2 = map(math.radians, [lat1, lat2, lon1, lon2])

	d_lat = lat2 - lat1
	d_lon = lon2 - lon1

	temp = (
		  math.sin(d_lat/2) ** 2
		+ math.cos(lat1)
		* math.cos(lat2)
		* math.sin(d_lon/2) ** 2
	)

	return 6373.0 * (2 * math.atan2(math.sqrt(temp), math.sqrt(1 - temp)))


def coord_to_square(lon, lat):
	'''
	| This function maps a coordinate pair to 
	| a x&y position in the greed.
	| x indexes longitude and y indexes latitude.
	'''
	x = math.floor( (lon-min_lon) / lon_step )
	y = math.floor( (lat-min_lat) / lat_step )
	x = x if x < DIVISIONS else x-1
	y = y if y < DIVISIONS else y-1
	return x, y


class Point: 
	def __init__(self, x, y): 
		self.x = x 
		self.y = y 
  

def on_segment(p, q, r):
	'''
	| Given three colinear points p, q, r, the function checks if
	| point q lies on line segment 'pr' 
	'''
	if ( (q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and 
		   (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))): 
		return True
	return False
  

def orientation(p, q, r):
	'''
	| To find the orientation of an ordered triplet (p,q,r) 
	| function returns the following values: 
	| 0 : Colinear points 
	| 1 : Clockwise points 
	| 2 : Counterclockwise 
	|  
	| See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/  
	| for details of below formula.  
	'''

	val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y)) 
	if (val > 0): 
		# Clockwise orientation 
		return 1

	elif (val < 0): 
		# Counterclockwise orientation 
		return 2

	else: 
		# Colinear orientation 
		return 0
  

def lines_intersect(line1, line2): 
	'''
	| The main function that returns true if 
	| the line segment 'line1' and 'line2' intersect. 
	'''

	p1 = Point(line1[0][0], line1[0][1])
	q1 = Point(line1[1][0], line1[1][1])
	p2 = Point(line2[0][0], line2[0][1])
	q2 = Point(line2[1][0], line2[1][1])

	# Find the 4 orientations required for  
	# the general and special cases 
	o1 = orientation(p1, q1, p2) 
	o2 = orientation(p1, q1, q2) 
	o3 = orientation(p2, q2, p1) 
	o4 = orientation(p2, q2, q1) 
  
	# General case 
	if ((o1 != o2) and (o3 != o4)): 
		return True
  
	# Special Cases 
  
	# p1 , q1 and p2 are colinear and p2 lies on segment p1q1 
	if ((o1 == 0) and on_segment(p1, p2, q1)): 
		return True
  
	# p1 , q1 and q2 are colinear and q2 lies on segment p1q1 
	if ((o2 == 0) and on_segment(p1, q2, q1)): 
		return True
  
	# p2 , q2 and p1 are colinear and p1 lies on segment p2q2 
	if ((o3 == 0) and on_segment(p2, p1, q2)): 
		return True
  
	# p2 , q2 and q1 are colinear and q1 lies on segment p2q2 
	if ((o4 == 0) and on_segment(p2, q1, q2)): 
		return True
  
	# If none of the cases 
	return False

def test_lines_intersect():
	'''
	| Small test suit for the 'lines_intersect' function
	'''
	line_a = [[2,  1], [5,  1]]
	line_b = [[3, -1], [3,  3]]
	line_c = [[2,  2], [5,  2]]
	line_d = [[2,  4], [4, -1]]
	line_e = [[4, -1], [2,  4]]

	line_f = [[-4, 5], [-1, 5]]
	line_g = [[-1, 5], [-1, 1]]
	line_h = [[-1, 1], [-4, 1]]
	line_i = [[-4, 1], [-4, 5]]
	line_j = [[-3, 3], [-5, 3]]

	print('Line A and B intercect      {}'.format('[OK]' if lines_intersect(line_a, line_b) else '[FAILED]'))
	print('Line A and C DONT intercect {}'.format('[OK]' if not lines_intersect(line_a, line_c) else '[FAILED]'))
	print('Line B and C intercect      {}'.format('[OK]' if lines_intersect(line_b, line_c) else '[FAILED]'))
	print('Line D and A intercect      {}'.format('[OK]' if lines_intersect(line_d, line_a) else '[FAILED]'))
	print('Line D and B intercect      {}'.format('[OK]' if lines_intersect(line_d, line_b) else '[FAILED]'))
	print('Line D and C intercect      {}'.format('[OK]' if lines_intersect(line_d, line_c) else '[FAILED]'))
	print('Line E and A intercect      {}'.format('[OK]' if lines_intersect(line_e, line_a) else '[FAILED]'))
	print('Line E and B intercect      {}'.format('[OK]' if lines_intersect(line_e, line_b) else '[FAILED]'))
	print('Line E and C intercect      {}'.format('[OK]' if lines_intersect(line_e, line_c) else '[FAILED]'))

	print('Line J and F DONT intercect {}'.format('[OK]' if not lines_intersect(line_j, line_f) else '[FAILED]'))
	print('Line J and G DONT intercect {}'.format('[OK]' if not lines_intersect(line_j, line_g) else '[FAILED]'))
	print('Line J and H DONT intercect {}'.format('[OK]' if not lines_intersect(line_j, line_h) else '[FAILED]'))
	print('Line J and I intercect      {}'.format('[OK]' if lines_intersect(line_j, line_i) else '[FAILED]'))


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

		if lines_intersect(origi_dest, right_edge) and not moved:
			if curr_x<de_x:
				curr_x += 1
				moved   = True

		if lines_intersect(origi_dest, lower_edge) and not moved:
			if curr_y>de_y:
				curr_y -= 1
				moved   = True

		if lines_intersect(origi_dest, lefty_edge) and not moved:
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

	# print(edge_info)

	return (no_way_out or no_way_in) and one_way


if __name__ == '__main__':
	'''
	| The value within which we will look for projected 
	| stops on the road network.
	'''
	THRESHOLD = 15

	'''
	| This script indexes road segments in a greed to make
	| computation faster, this is the number of divisions
	| to make in one dimension, meaning that the whole
	| road will be divided into DIVISIONS^2 squares.
	'''
	DIVISIONS = 100

	file = open('data/network.json', 'r')
	net  = json.load(file)
	file.close() 

	road_net = nx.readwrite.json_graph.adjacency_graph(net)

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
		
	bounds = [[{} for i in range(DIVISIONS)] for i in range(DIVISIONS)]
	lat_step = (max_lat-min_lat)/DIVISIONS
	lon_step = (max_lon-min_lon)/DIVISIONS

	stops_df = pd.read_csv('data/carris_gtfs/stops.txt', sep=',', decimal='.')
	route_df = pd.read_csv('data/carris_gtfs/shapes.txt', sep=',', decimal='.')
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

	for x in range(DIVISIONS):
		for y in range(DIVISIONS):
			bounds[x][y] = {
				'max_lat': min_lat + (y+1)*lat_step,
				'min_lat': min_lat + y*lat_step,
				'max_lon': min_lon + (x+1)*lon_step,
				'min_lon': min_lon + x*lon_step,
				'links':   []
			}

	for index, adjacencies in enumerate(net['adjacency']):
		print_progress_bar(index, len(net['adjacency']), prefix='[MAPPING] 1/2')
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

	print_progress_bar(len(net['adjacency']), len(net['adjacency']), prefix='[MAPPING] 1/2')

	stop_mappings = []
	for index, stop in enumerate(stops):
		'''
		| Now we well actually project the stations on the road.
		| The whole process of sorting the segments in squares is
		| to make this faster. Because we have the squares we can 
		| project the station just in the vacinity of the station
		| instead of the whole network.
		'''

		print_progress_bar(index, len(stops), prefix='[PROJECT] 2/2')

		lat = stop['lat']
		lon = stop['lon']
		stp = [lon, lat]
		pjs = [] 

		stp_x, stp_y = coord_to_square(lon, lat)
		candidate_coords = [[stp_x, stp_y]]
		
		if stp_x > 0:
			candidate_coords.append([stp_x-1, stp_y])

		if stp_y > 0:
			candidate_coords.append([stp_x, stp_y-1])

		if stp_x < DIVISIONS-1:
			candidate_coords.append([stp_x+1, stp_y])
		
		if stp_y < DIVISIONS-1:
			candidate_coords.append([stp_x, stp_y+1])

		if stp_x>0 and stp_y>0:
			candidate_coords.append([stp_x-1, stp_y-1])

		if stp_x<DIVISIONS-1 and stp_y<DIVISIONS-1:
			candidate_coords.append([stp_x+1, stp_y+1])

		if stp_x>0 and stp_y<DIVISIONS-1:
			candidate_coords.append([stp_x-1, stp_y+1])

		if stp_x<DIVISIONS-1 and stp_y>0:
			candidate_coords.append([stp_x+1, stp_y-1])

		added_thresh = 0
		while len(pjs) == 0:
			for candidate in candidate_coords:
				for segment in bounds[candidate[0]][candidate[1]]['links']:
					closest  = closest_point(
						segment['segment'][0], 
						segment['segment'][1], 
						stp
					)
					# edge_info = road_net.get_edge_data(
					# 	segment['origin_id'],
					# 	segment['destin_id'],
					# 	segment['key']
					# )
					# print('Projecting stop {} -> {}'.format(stop['stop_id'], edge_info))
					distance = haversine_distance(stp, closest)*1000
					if distance < THRESHOLD+added_thresh and not dead_end_edge(road_net, segment):
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

		#filtered_pjs.sort(key=lambda item: item['distance'])

		stop_mappings.append({
			'stop_id':  stop['stop_id'],
			'point':    stp,
			'mappings': final_pjs#filtered_pjs[:2]
		})

	print_progress_bar(len(stops), len(stops), prefix='[PROJECT] 2/2')

	with open('data/stop_mappings.json', 'w') as json_file:
		json.dump(stop_mappings, json_file, indent=4)