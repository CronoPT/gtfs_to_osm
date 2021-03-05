'''
| This script will take the chosen projection of each GTFS
| stop and make them a part of the road network.
'''


import networkx as nx
import math
import numpy as np
import utils.geometric_utils
import utils.json_utils
import configs


def compute_distance_to_or_and_de(point, coords):
	'''
	| This function will see where the 'point' belongs
	| in the geometry (line) defined by the sequence
	| of coordinate pairs in 'coords'. This function
	| will also compute the distance in the road between 
	| the origin of the line and the point and between
	| the point and the destiny of the line as well as 
	| the lines (geometries) that connect the origin to 
	| the new point and the new point to the destiny. 
	'''

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
			if utils.geometric_utils.point_belongs_to_line(point, line):
				distance_to_or += utils.geometric_utils.haversine_distance(prev,  point)
				distance_to_de += utils.geometric_utils.haversine_distance(point, curr)
				or_geometry.append(point)
				de_geometry.extend((point, curr))
				point_reached = True
			else:
				if not point_reached:
					distance_to_or += utils.geometric_utils.haversine_distance(prev, curr)
					or_geometry.append(curr)	
				else:
					distance_to_de += utils.geometric_utils.haversine_distance(prev, curr)
					de_geometry.append(curr)

		prev = curr

	return distance_to_or, distance_to_de, or_geometry, de_geometry


def update_observer_edge(new_edge, observer):
	'''
	| If observer shares some point with the newly
	| introduced edge, the observer must be updated.
	'''

	if observer['origin_id'] == new_edge['origin']:
		observer['destin_id'] = new_edge['destin']
	elif observer['destin_id'] == new_edge['destin']:
		observer['origin_id'] = new_edge['origin']


def insert_point_as_node(net, node_id, point_info, observers=[]):
	'''
	| This function makes a random point on the road network a node
	| in the network so that we can compute shortest path between
	| that point and others.
	| @params:
	|	net        - Required : the road network (nx.MultiDiGraph)
	| 	node_id    - Required : the node id to give to the point in 
	| the network (Int)
	|	point_info - Required : a dicionairy containing information 
	| about where to place the new node, between which nodes, in
	| which edge (Dict)  
	|	observers  - Optional : a list of other point_infos to be 
	| updated, essentially you will be deviding some edge in two
	| by placing a new node in the middle, this makes it so that
	| the pair of nodes between which other points are to be
	| introduced will change (List)
	'''

	edge_to_intersect = net.get_edge_data(
		point_info['origin_id'], 
		point_info['destin_id'],
		key = point_info['key']  
	)

	keep_attributes = {key:value for key, value in edge_to_intersect.items() if key not in [
		'length', 'geometry', 'origin', 'destin'
	]}

	point = point_info['point']
	origin_point_item = net.nodes[point_info['origin_id']] 
	destin_point_item = net.nodes[point_info['destin_id']]

	origin_point = [origin_point_item['x'], origin_point_item['y']]
	destin_point = [destin_point_item['x'], destin_point_item['y']]

	distance_to_or, distance_to_de = 0, 0
	or_geometry, de_geometry = None, None
	if 'geometry' in edge_to_intersect:
		res = compute_distance_to_or_and_de(
			point_info['point'], edge_to_intersect['geometry']
		)
		distance_to_or = res[0]
		distance_to_de = res[1]
		or_geometry    = res[2]
		de_geometry    = res[3]
	else:
		line = [origin_point, destin_point]

		if not utils.geometric_utils.point_belongs_to_line(point, line):
			raise Exception('You gave me something wrong')

		distance_to_or = utils.geometric_utils.haversine_distance(
			origin_point,
			point_info['point']
		)

		distance_to_de = utils.geometric_utils.haversine_distance(
			point_info['point'],
			destin_point
		)

	if or_geometry==None and de_geometry==None:
		
		net.add_edge(
			point_info['origin_id'],
			node_id,
			**{
				'length': distance_to_or,
				'origin': point_info['origin_id'],
				'destin': node_id
			}, **keep_attributes
		)

		net.add_edge(
			node_id,
			point_info['destin_id'],
			**{
				'length': distance_to_de,
				'origin': node_id,
				'destin': point_info['destin_id']
			}, **keep_attributes
		)
	else:
		net.add_edge(
			point_info['origin_id'],
			node_id,
			**{
				'length': distance_to_or,
				'origin': point_info['origin_id'],
				'destin': node_id,
				'geometry': or_geometry
			}, **keep_attributes
		)
		net.add_edge(
			node_id,
			point_info['destin_id'],
			**{
				'length': distance_to_de,
				'origin': node_id,
				'destin': point_info['destin_id'],
				'geometry': de_geometry
			}, **keep_attributes
		)

	for observer in observers:
		edge_to_keep = net.get_edge_data(
			observer['origin_id'], 
			observer['destin_id'],
			key = observer['key']  
		)
		if edge_to_keep == None:
			new_edge_1 = net.get_edge_data(
				point_info['origin_id'],
				node_id,
				key=0
			)
			new_edge_2 = net.get_edge_data(
				node_id,
				point_info['destin_id'],
				key=0
			)
			update_observer_edge(new_edge_1, observer)
			update_observer_edge(new_edge_2, observer)


if __name__ == '__main__':
 
	stop_points = utils.json_utils.read_json_object(configs.FINAL_MAPPINGS)
	route_paths = utils.json_utils.read_json_object(configs.ROUTES_STOP_SEQUENCE)
	road_net = utils.json_utils.read_network_json(configs.TWEAKED_NETWORK)

	for i in range(len(stop_points)):
		node_id = stop_points[i]['stop_id']
		insert_point_as_node(road_net, node_id, stop_points[i], observers=stop_points[i+1:])

	for stop_item in stop_points:
		stop_id = stop_item['stop_id']
		road_net.nodes[stop_id]['x'] = stop_item['point'][0]
		road_net.nodes[stop_id]['y'] = stop_item['point'][1]

	routes = []
	for route_info in route_paths:
		stops  = route_info['stops']
		origin = None
		route  = []
		for index, destin in enumerate(stops):
			if origin != None:
				path = nx.algorithms.shortest_paths.weighted.dijkstra_path(
					road_net, 
					source=origin, 
					target=destin,
					weight='length'
				)

				path_in_coordinates = []
				origin_node = None
				for index, destin_node in enumerate(path):
					if origin_node != None:
						edge_info = road_net.get_edge_data(origin_node, destin_node)

						best_edge = None
						best_cost = np.inf
						for key, edge_data in edge_info.items():
							length = edge_data['length']
							if length < best_cost:
								best_cost = length
								best_edge = edge_data 

						if 'geometry' in best_edge:
							path_in_coordinates.extend(best_edge['geometry'])
						else:
							path_in_coordinates.extend([
								[road_net.nodes[origin_node]['x'], road_net.nodes[origin_node]['y']],
								[road_net.nodes[destin_node]['x'], road_net.nodes[destin_node]['y']]
							])
					origin_node = destin_node

				if index==0:
					route.extend(path_in_coordinates)
				else:
					route.extend(path_in_coordinates[1:])

			origin = destin
		
		routes.append(route)

	utils.json_utils.write_geojson_lines(
		configs.ROUTE_SHAPES, 
		routes
	)
	utils.json_utils.write_geojson_points(
		configs.STOP_LOCATIONS,
		[item['point'] for item in stop_points]
	)
