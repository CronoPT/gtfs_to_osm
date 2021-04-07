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

	nodes_to_remove = []
	for pair in v_outs:
		edge = network.get_edge_data(pair[0], pair[1])[0]

		# if network.get_edge_data(pair[1], pair[0]) != None:
		# 	print('Warning2') 

		edge_geo  = []

		new_id = edge['id'] if edge['id'] not in replacements else replacements[edge['id']]

		if 'geometry' in v_outs:
			edge_geo = edge['geometry']
		else:
			fleeting_geo = [
				[network.nodes[v]['x'], network.nodes[v]['y']],
				[network.nodes[new_id]['x'], network.nodes[new_id]['y']]
			]

		new_edge_geo = []
		new_edge_geo.extend(fleeting_geo)
		new_edge_geo.extend(edge_geo)
		new_edge_geo = filter_duplicates(new_edge_geo)

		network.add_edge(u, new_id, **{
			'geometry': new_edge_geo,
			'length': edge['length']+fleeting['length'],
			'id': new_id
		})

		replacements[v] = u
		for og, rep in replacements.items():
			if rep == v:
				replacements[og] = u
		nodes_to_remove.append(v)

	for n in nodes_to_remove:
		network.remove_node(n)


if __name__ == '__main__':
	network  = utils.json_utils.read_network_json(configs.NETWORK_WITH_STOPS)
	routes   = utils.json_utils.read_json_object(configs.ROUTES_STOP_SEQUENCE)
	stop_points = utils.json_utils.read_json_object(configs.FINAL_MAPPINGS)
	route_df = pd.read_csv('data/PercursosOutubro2019.csv', sep=';', decimal=',', low_memory=False)

	stops = list(route_df['cod_paragem'].unique())
	valid = list(route_df['cod_paragem'].unique())

	replacements = {}

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