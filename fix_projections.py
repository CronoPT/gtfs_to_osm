'''
| This script implements the algorithm in Vuurstaek et al.
| It takes the candidate projections for each GTFS stop
| and choses the best one for each stop.
'''


import pandas as pd
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import numpy as np
import math
import copy
import utils.general_utils
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

	DEL['nodes'].append(node_id)
	net.add_node(node_id, **{
		'id': node_id,
		'x':  point_info['point'][0],
		'y':  point_info['point'][1]
	})

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

	ADD['links'].append({
		'origin': point_info['origin_id'],
		'destin': point_info['destin_id'],
		'attr_dict': net.get_edge_data(
			point_info['origin_id'], 
			point_info['destin_id'],
			key = point_info['key']  
		)
	})
	ADD['links'][-1]['attr_dict']['key'] = point_info['key']
	net.remove_edge(
		point_info['origin_id'], 
		point_info['destin_id'],
		key = point_info['key']
	)

	DEL['links'].append((point_info['origin_id'], node_id))
	DEL['links'].append((node_id, point_info['destin_id']))
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


def compute_distance_on_road_between(road_net, origin_point, destin_point):
	'''
	| The idea of this function is to compute the distance between 
	| two arbitrary points in the road network. They may not be nodes
	| but a point in the middle of a road segment, hence the trickyness. 
	'''

	global DEL
	global ADD

	insert_point_as_node(road_net, 0, origin_point, observers=[destin_point])
	insert_point_as_node(road_net, 1, destin_point)

	distance = 0

	try:
		distance = 	nx.algorithms.shortest_paths.weighted.dijkstra_path_length(
			road_net, 
			source=0, 
			target=1,
			weight='length'
		)
	except nx.exception.NetworkXNoPath:
		distance = -1

	for link in ADD['links']:
		road_net.add_edge(link['origin'], link['destin'], **link['attr_dict'])
	for link in DEL['links']:
		road_net.remove_edge(link[0], link[1])
	for node in DEL['nodes']:
		road_net.remove_node(node)
	DEL = {'nodes': [], 'links': []}
	ADD = {'nodes': [], 'links': []}
	
	return distance


def handle_triplets(Gg, Gp, gp_mappings):
	
	global _removed_one

	for node in Gg.nodes():
		in_deg  = Gg.in_degree(node)
		out_deg = Gg.out_degree(node)
		this_mappings = gp_mappings[node]

		if in_deg==1 and out_deg==1 and len(this_mappings)>1:
			in_edge  = Gg.in_edges(node)
			out_edge = Gg.out_edges(node)

			before_node = [u for u, _ in in_edge ][0]
			after_node  = [v for _, v in out_edge][0]

			before_mappings = gp_mappings[before_node] 
			after_mappings  = gp_mappings[after_node]

			total_paths = len(before_mappings)*len(after_mappings)
			path = []
			path_cost = 0
			path_counting = {p_node:0 for p_node in this_mappings} 
			for before_map in before_mappings:
				for after_map in after_mappings:

					min_cost = np.inf
					min_cost_path = []
					for proj in this_mappings:
						
						cost = Gp.get_edge_data(before_map, proj)['length'] + \
						       Gp.get_edge_data(proj, after_map)['length']

						if cost < min_cost:
							min_cost_path = [before_map, proj, after_map]
							min_cost = cost	

					path = min_cost_path
					path_cost = min_cost

					for p_node in this_mappings:
						if p_node in path:
							path_counting[p_node] += 1

			for p_node, count in path_counting.items():
				if count==0:
					Gp.remove_node(p_node)
					gp_mappings[node].remove(p_node)
					_removed_one = True
				elif count==total_paths:
					[Gp.remove_node(n) for n in gp_mappings[node] if n != p_node]
					gp_mappings[node] = [p_node]
					_removed_one = True
					break


def is_sink(G, node):
	out_deg = G.out_degree(node)
	return out_deg==0


def is_source(G, node):
	in_deg = G.in_degree(node)
	return in_deg==0


def is_assigned(node, mappings):
	return len(mappings[node])==1


def has_more_one_in(G, node):
	in_deg = G.in_degree(node)
	return in_deg > 1


def has_more_one_out(G, node):
	out_deg = G.out_degree(node)
	return out_deg > 1


def is_cycle_breaker(G, node):
	return 'breaker' in G.nodes[node]


def is_boundary(Gg, node, mappings):
	return is_sink(Gg, node) or           \
	       is_source(Gg, node) or         \
		   is_assigned(node, mappings) or \
		   has_more_one_in(Gg, node) or   \
		   has_more_one_out(Gg, node) or  \
		   is_cycle_breaker(Gg, node)	   
	

def handle_non_bifurcating(Gg, Gp, gp_mappings):
	
	global _removed_one

	dfs_edges = list(nx.dfs_edges(Gg))

	prev_destin = None
	sequences   = [] 
	curr_sequence = []
	for index, dfs_edge in enumerate(dfs_edges):
		curr_origin = dfs_edge[0]
		curr_destin = dfs_edge[1]

		if curr_origin!=prev_destin and prev_destin!=None:
			curr_sequence.append(prev_destin)
			sequences.append(curr_sequence)
			curr_sequence = []
			
		if is_boundary(Gg, curr_origin, gp_mappings):
			curr_sequence.append(curr_origin)
			sequences.append(curr_sequence)
			curr_sequence = []
		else:
			curr_sequence.append(curr_origin)

		prev_destin = curr_destin

	sequences = [s for s in sequences if len(s)>2]
	
	for sequence in sequences:
		origin = sequence[0]
		destin = sequence[-1]
		middle = sequence[1:-1]

		origin_mappings = gp_mappings[origin]
		destin_mappings = gp_mappings[destin]

		path_counts = {n:{m:0 for m in gp_mappings[n]} for n in middle} 
		total_paths = len(origin_mappings)*len(destin_mappings)
		for or_map in origin_mappings:
			for de_map in destin_mappings:
				try:
					path = nx.algorithms.shortest_paths.weighted.dijkstra_path(
						Gp, 
						source=or_map, 
						target=de_map,
						weight='length'
					)

					for index, p_node in enumerate(path[1:-1]):
						path_counts[middle[index]][p_node] += 1
				
				except nx.exception.NetworkXNoPath:
					total_paths -= 1

		for node, countings in path_counts.items():
			for p_node, count in countings.items():
				if count == 0:
					Gp.remove_node(p_node)
					gp_mappings[node].remove(p_node)
					_removed_one = True
				elif count == total_paths:
					[Gp.remove_node(n) for n in gp_mappings[node] if n != p_node]
					gp_mappings[node] = [p_node]
					_removed_one = True
					break


def is_component_boundary(Gg, node, gp_mappings):
	return is_sink(Gg, node) or          \
	       is_source(Gg, node) or        \
		   is_cycle_breaker(Gg, node) or \
		   is_assigned(node, gp_mappings) 


def decompose(Gg, Gp, gp_mappings):
	components = []
	
	sources = [node for node in Gg.nodes if is_source(Gg, node)]
	
	for source in sources:
		queue = []
		queue_reserve  = []
		used_bounds    = []
		explored_nodes = []

		queue.append(source)
		components_left = True
		while components_left:
			curr_component  = nx.DiGraph()
			components_left = False
			while len(queue)>0:
				node  = queue.pop(0)
				edges = Gg.out_edges(node)

				for edge in edges:
					origin = edge[0]
					destin = edge[1]
					if not (is_component_boundary(Gg, origin, gp_mappings) and \
							is_component_boundary(Gg, destin, gp_mappings)):

						curr_component.add_edge(*edge) #further breaks components into more wCCs

				new_nodes = [v for _, v in edges]
				for new_node in new_nodes:
					if is_component_boundary(Gg, new_node, gp_mappings) and \
					new_node not in used_bounds:
						queue_reserve.append(new_node)
						used_bounds.append(new_node)
					elif new_node not in explored_nodes:
						queue.append(new_node)
						explored_nodes.append(new_node)

			queue = queue_reserve
			queue_reserve = []
			components_left = len(queue)>0
			components.append(curr_component)

	broken_components = []
	for component in components:
		wccs = list(nx.weakly_connected_components(component))
		broken_components.extend([Gg.subgraph(wcc) for wcc in wccs])

	return broken_components


def mark_as_cycle_breaker(G, node):
	G.nodes[node]['breaker'] = True


def handle_stars(Gg, Gp, gp_mappings):
	
	global _removed_one

	for node in Gg.nodes():
		in_deg  = Gg.in_degree(node)
		out_deg = Gg.out_degree(node)
		this_mappings = gp_mappings[node]

		if ((in_deg==1 and out_deg>1) or (in_deg>1 and out_deg==1)) and \
		   len(gp_mappings[node])>1:

			before_nodes = [u for u, _ in Gg.in_edges(node)]
			after_nodes  = [v for _, v in Gg.out_edges(node)]

			path_count  = {n:0 for n in this_mappings}
			total_paths = 0
			for before_node in before_nodes:
				for after_node in after_nodes:
					origin_mappings = gp_mappings[before_node]
					destin_mappings = gp_mappings[after_node]

					total_paths += len(origin_mappings)*len(destin_mappings)

					
					for or_map in origin_mappings:
						for de_map in destin_mappings:
							min_path = None
							min_path_cost = np.inf
							for this_map in this_mappings:
								path_cost  = Gp.get_edge_data(or_map, this_map)['length']
								path_cost += Gp.get_edge_data(this_map, de_map)['length']
								if path_cost < min_path_cost:
									min_path = this_map
									min_path_cost = path_cost
							

							path_count[min_path] += 1


			for p_node, count in path_count.items():
				if count==0:
					Gp.remove_node(p_node)
					gp_mappings[node].remove(p_node)
					_removed_one = True
				elif count==total_paths: 
					[Gp.remove_node(n) for n in gp_mappings[node] if n != p_node]
					gp_mappings[node] = [p_node]
					_removed_one = True
					break


def assign_projections(C, Gg, Gp, gp_mappings):

	global _total_combinations
	global _combinations_tried

	possible_assigns = 1
	for node in C.nodes:
		possible_assigns *= len(gp_mappings[node])

	if possible_assigns==1: #everything already assigned
		return

	lengths = {} 
	for index, node in enumerate(C.nodes):
		lengths[node] = {
			'index': index,
			'candidates': gp_mappings[node]
		}

	best_assign = None
	best_cost   = np.inf
	assign_options = itertools.product(*[range(len(item['candidates'])) for _, item in lengths.items()])
	for assign in assign_options:
		this_cost = 0
		for edge in C.edges:
			origin = edge[0]
			destin = edge[1]

			p_origin = gp_mappings[origin][assign[lengths[origin]['index']]]
			p_destin = gp_mappings[destin][assign[lengths[destin]['index']]]

			this_cost += Gp.get_edge_data(p_origin, p_destin)['length']

		if this_cost < best_cost:
			best_cost   = this_cost
			best_assign = assign

	for node in C.nodes:
		p_node = gp_mappings[node][best_assign[lengths[node]['index']]]
		gp_mappings[node] = [p_node]


def fix_remaining(Gg, Gp, gp_mappings):

	for node in Gg.nodes:
		if len(gp_mappings[node])>1:
			in_edges  = Gg.in_edges(node)
			out_edges = Gg.out_edges(node)

			best_cost = np.inf
			best_opt  = []
			for p_option in gp_mappings[node]: 
				this_cost = 0

				for in_edge in in_edges:
					origin = in_edge[0]
					info = Gp.get_edge_data(
						gp_mappings[origin][0],
						p_option
					)
					this_cost += info['length']

				for out_edge in out_edges:
					destin = out_edge[1]
					info = Gp.get_edge_data(
						p_option,
						gp_mappings[destin][0]
					)
					this_cost += info['length']

				if this_cost < best_cost:
					best_cost = this_cost
					best_opt  = p_option
			
			gp_mappings[node] = [best_opt]


if __name__ == '__main__':

	'''
	| We will be introducing points in the network iterativelly,
	| to avoid having to read the network from a json file
	| at every iteration, this structures will allow us to track
	| any changes made to the network and undo them once we are
	| done, making the whole script run much faster.
	'''
	DEL = {'nodes': [], 'links': []}
	ADD = {'nodes': [], 'links': []}

	road_net = utils.json_utils.read_network_json(configs.TWEAKED_NETWORK)

	stops_df = pd.read_csv('data/carris_gtfs/stops.txt', sep=',', decimal='.')
	route_df = pd.read_csv('data/PercursosOutubro2019.csv', sep=';', decimal=',', low_memory=False)

	route_ids = []
	stop_ids  = []
	for index, row in route_df.iterrows():
		utils.general_utils.print_progress_bar(
			index, 
			route_df.shape[0], 
			prefix='[STOP CHECK]    1/4'
		)
		route_ids.append('{}{}{}'.format(row['carreira'], row['sentido'], row['variante']))
		stop_ids.append(int(row['cod_paragem']))
	utils.general_utils.print_progress_bar(
		route_df.shape[0], 
		route_df.shape[0], 
		prefix='[STOP CHECK]    1/4'
	)
	route_df['shape_id'] = route_ids 
	route_df['stop_id']  = stop_ids

	nodes = {}
	graph_edges = []
	_purged = []
	_g_purg = nx.DiGraph()
	route_paths = []
	for index, shape in enumerate(route_df['shape_id'].unique()):
		utils.general_utils.print_progress_bar(
			index, 
			len(route_df['shape_id'].unique()), 
			prefix='[GRAPH BUILD]   2/4'
		)

		sequence = route_df[ route_df['shape_id']==shape ]
		sequence.sort_values('n_ordem')

		route_path = {
			'route_id': shape,
			'stops': []
		}
		route_paths.append(route_path)

		prev = None
		for _, row in sequence.iterrows():
			lon = row['longitude']
			lat = row['latitude']
			stop_id = row['stop_id']
			coords  = (lon, lat)
			curr    = None 

			route_path['stops'].append(stop_id)

			if stop_id not in nodes:
				curr = {
					'stop_id': stop_id,
					'lon': lon,
					'lat': lat,
				}
				nodes[stop_id] = curr
			else:
				curr = nodes[stop_id]

			if prev != None:
				graph_edges.append((prev['stop_id'], curr['stop_id']))
			
				if prev['stop_id'] not in _purged and curr['stop_id'] not in _purged:
						try:
							distance = nx.algorithms.shortest_paths.weighted.dijkstra_path_length(
								_g_purg, 
								source=curr['stop_id'], 
								target=prev['stop_id'],
							)
							nodes[stop_id]['breaker'] = True
							_purged.append(curr['stop_id'])
						except nx.exception.NetworkXNoPath:
							_g_purg.add_edge(prev['stop_id'], curr['stop_id'])
						except nx.exception.NodeNotFound:
							_g_purg.add_edge(prev['stop_id'], curr['stop_id'])
			prev = curr
	
	utils.general_utils.print_progress_bar(
		len(route_df['shape_id'].unique()), 
		len(route_df['shape_id'].unique()), 
		prefix='[GRAPH BUILD]   2/4'
	)

	graph_nodes = []
	for _, obj in nodes.items():
		if 'breaker' in obj:
			graph_nodes.append((obj['stop_id'], {
				'lon': obj['lon'],
				'lat': obj['lat'],
				'breaker': True
			}))
		else:
			graph_nodes.append((obj['stop_id'], {
				'lon': obj['lon'],
				'lat': obj['lat'],
			}))

	Gg = nx.DiGraph()
	Gg.add_nodes_from(graph_nodes)
	Gg.add_edges_from(graph_edges)

	mappings = utils.json_utils.read_json_object(configs.CANDIDATE_MAPPINGS)

	p_graph_nodes = []
	p_graph_edges = []
	_impossible_paths = []
	_impossible_paths_assetion = []

	gp_mappings = {}
	counter     = 0
	for map_item in mappings:
		stop_id = map_item['stop_id']
		gp_mappings[stop_id] = []
		for projection in map_item['mappings']:
			gp_mappings[stop_id].append(counter)
			p_graph_nodes.append((counter ,{
				'lon': projection['point'][0],
				'lat': projection['point'][1]
			}))

			counter += 1

	node_lon = nx.get_node_attributes(Gg, 'lon')
	node_lat = nx.get_node_attributes(Gg, 'lat')
	for index, edge in enumerate(Gg.edges):
		utils.general_utils.print_progress_bar(
			index, 
			len(Gg.edges), 
			prefix='[P GRAPH BUILD] 3/4'
		)

		origin_stop_id = edge[0]
		destin_stop_id = edge[1]

		origin_map_item = list(filter(lambda item: item['stop_id']==origin_stop_id, mappings))[0]
		destin_map_item = list(filter(lambda item: item['stop_id']==destin_stop_id, mappings))[0]

		for or_index, origin_projection in enumerate(origin_map_item['mappings']):
			for de_index, destin_projection in enumerate(destin_map_item['mappings']):
				origin_point = origin_projection['point']
				destin_point = destin_projection['point']

				tuple_origin_point = origin_point[0], origin_point[1]
				tuple_destin_point = destin_point[0], destin_point[1]

				origin_node_id = gp_mappings[origin_stop_id][or_index]
				destin_node_id = gp_mappings[destin_stop_id][de_index]

				length = compute_distance_on_road_between(
					road_net,
					copy.copy(origin_projection), 
					copy.copy(destin_projection)
				)

				if length != -1:
					p_graph_edges.append((origin_node_id, destin_node_id, {
						'length': length
					}))
				else:
					p_graph_edges.append((origin_node_id, destin_node_id, {
						'length': np.inf
					}))
					_impossible_paths.append([tuple_origin_point, tuple_destin_point])	

	utils.general_utils.print_progress_bar(
		len(Gg.edges), 
		len(Gg.edges), 
		prefix='[P GRAPH BUILD] 3/4'
	)
				
	Gp = nx.DiGraph()
	Gp.add_nodes_from(p_graph_nodes)
	Gp.add_edges_from(p_graph_edges)

	_removed_one = True
	cycles = 0
	while _removed_one:
		_removed_one = False

		handle_triplets(Gg, Gp, gp_mappings)

		handle_non_bifurcating(Gg, Gp, gp_mappings)

		handle_stars(Gg, Gp, gp_mappings)
		cycles += 1

	components = decompose(Gg, Gp, gp_mappings)

	for index, component in enumerate(components):
		utils.general_utils.print_progress_bar(
			index, 
			len(components), 
			prefix='[ASSIGNMENT]    4/4'
		)
		assign_projections(component, Gg, Gp, gp_mappings)
	utils.general_utils.print_progress_bar(
		len(components), 
		len(components), 
		prefix='[ASSIGNMENT]    4/4'
	)

	fix_remaining(Gg, Gp, gp_mappings)

	final_mappings = []
	for stop, mapping in gp_mappings.items():

		stop_item = list(filter(lambda item: item['stop_id']==stop, mappings))[0]
		point = [Gp.nodes[mapping[0]]['lon'], Gp.nodes[mapping[0]]['lat']]
		proj_item = list(filter(lambda item: item['point']==point, stop_item['mappings']))[0]

		final_mappings.append({
			'stop_id': stop,
			'point': [
				Gp.nodes[mapping[0]]['lon'],
				Gp.nodes[mapping[0]]['lat']
			],
			'origin_id': proj_item['origin_id'],
			'destin_id': proj_item['destin_id'],
			'key': proj_item['key']
		})

	utils.json_utils.write_json_object(configs.FINAL_MAPPINGS, final_mappings)
	utils.json_utils.write_json_object(configs.ROUTES_STOP_SEQUENCE, route_paths)
