import pandas as pd
import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math

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


def point_belongs_to_line(point, line):
	point = np.array(point)
	line_point_1 = np.array(line[0])
	line_point_2 = np.array(line[1])
	
	distance_to_1 = np.linalg.norm(point-line_point_1)
	distance_to_2 = np.linalg.norm(point-line_point_2)
	length = np.linalg.norm(line_point_1-line_point_2)

	return distance_to_1+distance_to_2 == length


def compute_distance_to_or_and_de(point, coords):
	distance_to_or = 0
	distance_to_de = 0
	or_geometry = []
	de_geometry = []
	point_reached  = False

	curr = None
	prev = None
	for coord in coords:
		curr = coord
		if prev != None:
			line = [prev, curr]
			if point_belongs_to_line(point, line):
				distance_to_or += haversine_distance(prev,  point)
				distance_to_de += haversine_distance(point, curr)
				or_geometry.append(point)
				de_geometry.extend((point, curr))
				point_reached = True
			else:
				if not point_reached:
					distance_to_or += haversine_distance(prev, curr)
					or_geometry.append(curr)	
				else:
					distance_to_de += haversine_distance(prev, curr)
					de_geometry.append(curr)

		prev = curr

	return distance_to_or, distance_to_de

def insert_point_as_node(net, node_id, point_info):
	edge_to_intersect = net.get_edge_data(
		point_info['origin_id'], 
		point_info['destin_id'],
		key = point_info['key']  
	)

	point = point_info['point']
	origin_point_item = net.nodes[point_info['origin_id']] 
	destin_point_item = net.nodes[point_info['destin_id']]

	origin_point = [origin_point_item['x'], origin_point_item['y']]
	destin_point = [destin_point_item['x'], destin_point_item['y']]

	net.add_node(node_id, attr_dict={
		'id': node_id,
		'x':  point_info['point'][0],
		'y':  point_info['point'][1]
	})

	distance_to_or, distance_to_de = 0, 0
	if 'geometry' in edge_to_intersect:
		distance_to_or, distance_to_de = compute_distance_to_or_and_de(
			point_info['point'], edge_to_intersect['geometry']
		)
	else:
		line_point_1 = net.node[point_info['origin_id']]
		line_point_2 = net.node[point_info['destin_id']]
		line = [line_point_1, line_point_2]
		if not point_belongs_to_line(point, line):
			raise Exception('You gave me something wrong')
		
		distance_to_or = haversine_distance(
			point_info['point'],
			origin_point
		)

		distance_to_de = haversine_distance(
			point_info['point'],
			destin_point
		)

	net.remove_edge(
		point_info['origin_id'], 
		point_info['destin_id'],
		key = point_info['key']
	)
	net.add_edge(
		point_info['origin_id'],
		node_id,
		attr_dict={
			'length': distance_to_or
		},
	)
	net.add_edge(
		node_id,
		point_info['destin_id'],
		attr_dict={
			'length': distance_to_de
		},
	)

def compute_distance_on_road_between(origin_point, destin_point):
	'''
	| The idea of this function is to compute the distance between 
	| two arbitrary points in the road network. They may not be nodes
	| but a point in the middle of a road segment, hence the trickyness 
	'''

	file = open('data/lisbon_net_data.json', 'r')
	network_json = json.load(file)
	file.close() 

	road_net = nx.readwrite.json_graph.adjacency_graph(network_json)

	origin_edge = road_net.get_edge_data(
		origin_point['origin_id'], 
		origin_point['destin_id']
	)

	destin_edge = road_net.get_edge_data(
		destin_point['origin_id'], 
		destin_point['destin_id']
	)

	insert_point_as_node(road_net, 0, origin_point)
	insert_point_as_node(road_net, 1, destin_point)

	return nx.algorithms.shortest_paths.weighted.dijkstra_path_length(
		road_net, 
		source=0, 
		target=1,
		weight='length'
	)

if __name__ == '__main__':
	origin_point = {
		'point': [
			-9.1287703,
			38.7678144
		],
		'origin_id': 5813875505,
		'destin_id': 247114466,
		'key': 0
	}

	destin_point = {
		'point': [
			-9.127845441836154,
			38.769005268764616
		],
		'origin_id': 2263219093,
		'destin_id': 247114702,
		'key': 0
	}

	a = compute_distance_on_road_between(origin_point, destin_point)
	print(a)

# if __name__ == '__main__':
# 	stops_df = pd.read_csv('data/carris_gtfs/stops.txt', sep=',', decimal='.')
# 	route_df = pd.read_csv('data/carris_gtfs/shapes.txt', sep=',', decimal='.')

# 	file = open('data/lisbon_net_data.json', 'r')
# 	net  = json.load(file)
# 	file.close() 

# 	stop_ids = []
# 	not_stop = []
# 	for index, row in route_df.iterrows():
# 		print_progress_bar(index, route_df.shape[0], prefix='[STOP CHECK]   ')

# 		lat = row['shape_pt_lat']
# 		lon = row['shape_pt_lon']
# 		res = stops_df.loc[(stops_df['stop_lat']==lat) & (stops_df['stop_lon']==lon)]
# 		if res.shape[0] > 0:
# 			stop_ids.append(res.iloc[0]['stop_id'])
# 		else:
# 			stop_ids.append('Not Stop')
# 			not_stop.append([lon, lat])
	
# 	print_progress_bar(route_df.shape[0], route_df.shape[0], prefix='[STOP CHECK]   ')

# 	route_df['stop_id'] = stop_ids
# 	route_df = route_df[route_df['stop_id']!='Not Stop']

# 	nodes = {}
# 	graph_edges = []

# 	for index, shape in enumerate(route_df['shape_id'].unique()):
# 		print_progress_bar(index, len(route_df['shape_id'].unique()), prefix='[GRAPH BUILD]  ')

# 		sequence = route_df[ route_df['shape_id']==shape ]
# 		sequence.sort_values('shape_pt_sequence')

# 		prev = None
# 		for _, row in sequence.iterrows():
# 			lon = row['shape_pt_lon']
# 			lat = row['shape_pt_lat']
# 			stop_id = row['stop_id']
# 			coords  = (lon, lat)
# 			curr    = None 

# 			if coords not in nodes:
# 				curr = {
# 					'id':  len(nodes),
# 					'lon': lon,
# 					'lat': lat,
# 					'stop_id': stop_id
# 				}
# 				nodes[coords] = curr
# 			else:
# 				curr = nodes[coords]

# 			if prev != None:
# 				graph_edges.append((prev['id'], curr['id']))

# 			prev = curr
	
# 	print_progress_bar(len(route_df['shape_id'].unique()), 
# 	                   len(route_df['shape_id'].unique()), 
# 					   prefix='[GRAPH BUILD]  ')

# 	graph_nodes = []
# 	for _, obj in nodes.items():
# 		graph_nodes.append((obj['id'], {
# 			'lon': obj['lon'],
# 			'lat': obj['lat'],
# 			'stop_id': obj['stop_id']
# 		}))

# 	Gg = nx.Graph()
# 	Gg.add_nodes_from(graph_nodes)
# 	Gg.add_edges_from(graph_edges)

# 	file = open('data/stop_mappings.json', 'r')
# 	mappings = json.load(file)
# 	file.close() 

# 	p_nodes = {}
# 	p_graph_edges = []

# 	node_lon = nx.get_node_attributes(Gg, 'lon')
# 	node_lat = nx.get_node_attributes(Gg, 'lat')
# 	node_stop_ids = nx.get_node_attributes(Gg, 'stop_id')
# 	for index, edge in enumerate(Gg.edges):
# 		print_progress_bar(index, len(Gg.edges), prefix='[P GRAPH BUILD]')

# 		origin = edge[0]
# 		destin = edge[1]

# 		origin_stop_id = node_stop_ids[origin]
# 		destin_stop_id = node_stop_ids[destin]

# 		origin_map_item = list(filter(lambda item: item['stop_id']==origin_stop_id, mappings))[0]
# 		destin_map_item = list(filter(lambda item: item['stop_id']==destin_stop_id, mappings))[0]

# 		for origin_projection in origin_map_item['mappings']:
# 			for destin_projection in destin_map_item['mappings']:
# 				origin_point = origin_projection['point']
# 				destin_point = destin_projection['point']

# 				tuple_origin_point = origin_point[0], origin_point[1]
# 				tuple_destin_point = destin_point[0], destin_point[1]

# 				if tuple_origin_point not in p_nodes:
# 					p_nodes[tuple_origin_point] = len(p_nodes)

# 				if tuple_destin_point not in p_nodes:
# 					p_nodes[tuple_destin_point] = len(p_nodes)
				
# 				origin_node_id = p_nodes[tuple_origin_point]
# 				destin_node_id = p_nodes[tuple_destin_point]

# 				p_graph_edges.append((origin_node_id, destin_node_id))

# 	print_progress_bar(len(Gg.edges), len(Gg.edges), prefix='[P GRAPH BUILD]')

# 	p_graph_nodes = []
# 	for coords, node_id in p_nodes.items():
# 		p_graph_nodes.append((node_id, {
# 			'lon': coords[0],
# 			'lat': coords[1]
# 		}))
				
# 	Gp = nx.Graph()
# 	Gp.add_nodes_from(p_graph_nodes)
# 	Gp.add_edges_from(p_graph_edges)

# 	nx.draw_networkx(Gg, pos={
# 		node: (obj['lon'], obj['lat'])
# 		for node, obj in graph_nodes
# 	}, with_labels=False, node_size=10)

# 	plt.show()
# 	plt.clf()

# 	nx.draw_networkx(Gp, pos={
# 		node: (obj['lon'], obj['lat'])
# 		for node, obj in p_graph_nodes
# 	}, with_labels=False, node_size=10)

# 	plt.show()