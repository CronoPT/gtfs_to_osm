from fuzzywuzzy import fuzz
from fuzzywuzzy import process

import networkx as nx
import pandas as pd
import osmnx as ox
import utils.json_utils
import utils.general_utils
import configs
import json

def filter_duplicates(lst):	
	res = []
	for i in lst:
		if i not in res:
			res.append(i)
	return res
	

def single_connection(network, u, v):
	v_ins = network.in_edges(v)
	for ori,_ in v_ins:
		if ori != u:
			return False
	return True


def merge_stops(network, u, v):
	global replacements

	v_outs   = network.out_edges(v)
	fleeting = network.get_edge_data(u, v, key=0) 

	fleeting_geo = []

	if 'geometry' in fleeting:
		fleeting_geo = fleeting['geometry']
	else:
		fleeting_geo = [
			[network.nodes[u]['x'], network.nodes[u]['y']],
			[network.nodes[v]['x'], network.nodes[v]['y']]
		]

	for pair in v_outs:
		edges = network.get_edge_data(pair[0], pair[1])
		for _, edge in edges.items():

			edge_geo  = []

			if 'geometry' in v_outs:
				edge_geo = edge['geometry']
			else:
				fleeting_geo = [
					[network.nodes[v]['x'], network.nodes[v]['y']],
					[network.nodes[edge['id']]['x'], network.nodes[edge['id']]['y']]
				]

			new_edge_geo = []
			new_edge_geo.extend(fleeting_geo)
			new_edge_geo.extend(edge_geo)
			new_edge_geo = filter_duplicates(new_edge_geo)

			network.add_edge(u, edge['id'], **{
				'geometry': new_edge_geo,
				'length': edge['length']+fleeting['length'],
				'id': edge['id']
			})

			replacements[v] = u
			for og, rep in replacements.items():
				if rep == v:
					replacements[og] = u

	network.remove_node(v)


if __name__ == '__main__':
	network  = utils.json_utils.read_network_json(configs.NETWORK_WITH_STOPS)
	routes   = utils.json_utils.read_json_object(configs.ROUTES_STOP_SEQUENCE)
	stop_points = utils.json_utils.read_json_object(configs.FINAL_MAPPINGS)
	route_df = pd.read_csv('data/PercursosOutubro2019.csv', sep=';', decimal=',', low_memory=False)

	stops = list(route_df['cod_paragem'].unique())
	valid = list(route_df['cod_paragem'].unique())

	replacements = {}

	# tweaked_network = utils.json_utils.read_network_json(configs.TWEAKED_NETWORK)

	# point0 = [38.706295, -9.143219]
	# point1 = [38.705911, -9.143187]
	# point2 = [38.705801, -9.143223]
	# point3 = [38.705678, -9.143217]
	# point4 = [38.705544, -9.143223]
	# point5 = [38.705358, -9.143273]

	# node0 = int(ox.distance.get_nearest_node(network, point0))
	# node1 = int(ox.distance.get_nearest_node(network, point1))
	# node2 = int(ox.distance.get_nearest_node(network, point2))
	# node3 = int(ox.distance.get_nearest_node(network, point3))
	# node4 = int(ox.distance.get_nearest_node(network, point4))
	# node5 = int(ox.distance.get_nearest_node(network, point5))

	# print(f'[DEST] Node 0 -> {node0}')
	# print(f'[STOP] Node 1 -> {node1}')
	# print(f'[STOP] Node 2 -> {node2}')
	# print(f'[STOP] Node 3 -> {node3}')
	# print(f'[STOP] Node 4 -> {node4}')
	# print(f'[ORIG] Node 5 -> {node5}')


	# print(network.get_edge_data(node1, node2))
	# print(network.get_edge_data(node2, node1))

	# print(network.get_edge_data(node2, node3))
	# print(network.get_edge_data(node3, node2))

	# print(network.get_edge_data(node3, node4))
	# print(network.get_edge_data(node4, node3))

	# print(network.get_edge_data(node4, node5))
	# print(network.get_edge_data(node5, node4))

	# print(tweaked_network.get_edge_data(node0, node5))
	# print(tweaked_network.get_edge_data(node5, node0))


	for i, u in enumerate(stops):
		for j, v in enumerate(stops):
			if u in valid and v in valid:
				utils.general_utils.print_progress_bar(((i)*len(stops))+(j+1), len(stops)**2)
				if (network.get_edge_data(u, v) != None) and single_connection(network, u, v):
					try:
						distance = nx.algorithms.shortest_paths.weighted.dijkstra_path_length(
							network, 
							source=u, 
							target=v,
							weight='length'
						)
					except nx.exception.NetworkXNoPath:
						continue
					if distance > 100:
						continue

					merge_stops(network, u, v)
					valid.remove(v)


	stop_points = [s for s in stop_points if s['stop_id'] not in replacements]
	for route in routes:
		route['stops'] = [int(s) if s not in replacements else int(replacements[s]) 
		                  for s in route['stops']]
		route['stops'] = filter_duplicates(route['stops'])

	utils.json_utils.write_json_object(configs.CLUSTERED_STOPS, stop_points)
	utils.json_utils.write_json_object(configs.CLUSTERED_ROUTES, routes)
	utils.json_utils.write_networkx_json(configs.FINAL_NETWORK, network)