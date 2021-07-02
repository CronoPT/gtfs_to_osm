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

	v_outs   = network.out_edges(v, data=True)
	fleeting = network.get_edge_data(u, v, key=0) 

	fleeting_geo = []

	if 'geometry' in fleeting:
		fleeting_geo.extend(fleeting['geometry'])
	else:
		fleeting_geo = [
			[network.nodes[u]['x'], network.nodes[u]['y']],
			[network.nodes[v]['x'], network.nodes[v]['y']]
		]

	for e_u, e_v, edge_data in v_outs:
		edge_geo  = []

		if 'geometry' in edge_data:
			edge_geo.extend(edge_data['geometry'])
		else:
			fleeting_geo = [
				[network.nodes[e_u]['x'], network.nodes[e_u]['y']],
				[network.nodes[e_v]['x'], network.nodes[e_v]['y']]
			]

		new_edge_geo = []
		new_edge_geo.extend(fleeting_geo)
		new_edge_geo.extend(edge_geo)
		new_edge_geo = filter_duplicates(new_edge_geo)

		network.add_edge(u, e_v, **{
			'geometry': new_edge_geo,
			'length': edge_data['length']+fleeting['length'],
			'id': e_v
		})

		replacements[v] = u
		for og, rep in replacements.items():
			if rep == v:
				replacements[og] = u

	network.remove_node(v)


def has_nothing_from_cluster(stop_out, cluster):
	for out in stop_out:
		if out in cluster:
			return False
	return True


def merge_stops_in_cluster(network, cluster):
	global replacements
	sequence = []
	while len(cluster) != 0:
		chosen = None
		for stop in cluster:
			stop_outs = [int(v) for _, v in network.out_edges(int(stop))]
			if has_nothing_from_cluster(stop_outs, cluster):
				chosen = stop
				break
		sequence.append(chosen)
		cluster.remove(chosen)

	for i in range(0, len(sequence)-1, 1):
		merge_stops(network, sequence[i+1], sequence[i])

	for i in sequence[:-1]:
		replacements[i] = sequence[-1]


def belongs_to_cluster(clusters, s):
	for index, cluster in enumerate(clusters):
		if s in cluster:
			return index
	return -1


def squash_clusters(clusters, i1, i2):
	c1 = clusters[i1]
	c2 = clusters[i2]
	clusters.remove(c1)
	clusters.remove(c2)
	c1.extend(c2)
	clusters.append(c1)


def cluster_stops(clusters, s1, s2):
	cluster1 = belongs_to_cluster(clusters, s1)
	cluster2 = belongs_to_cluster(clusters, s2)

	if cluster1!=-1 and cluster2!=-1:
		squash_clusters(clusters, cluster1, cluster2)
	elif cluster1!=-1 and cluster2==-1:
		clusters[cluster1].append(s2)
	elif cluster1==-1 and cluster2!=-1:
		clusters[cluster2].append(s1)
	else:
		clusters.append([s1, s2])


if __name__ == '__main__':
	network  = utils.json_utils.read_network_json(configs.NETWORK_WITH_STOPS)
	routes   = utils.json_utils.read_json_object(configs.ROUTES_STOP_SEQUENCE)
	stop_points = utils.json_utils.read_json_object(configs.FINAL_MAPPINGS)
	route_df = pd.read_csv('data/PercursosOutubro2019.csv', sep=';', decimal=',', low_memory=False)

	stops = list(route_df['cod_paragem'].unique())

	print(f'Stops before clustering: {len(stop_points)}')

	replacements = {}
	clusters = []

	for i, u in enumerate(stops):
		for j, v in enumerate(stops):
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

				cluster_stops(clusters, u, v)

	for cluster in clusters:
		merge_stops_in_cluster(network, cluster)

	stop_points = [s for s in stop_points if s['stop_id'] not in replacements]
	for route in routes:
		route['stops'] = [int(s) if s not in replacements else int(replacements[s]) 
		                  for s in route['stops']]
		route['stops'] = filter_duplicates(route['stops'])

	final = {}
	for key, map in replacements.items():
		final[int(key)] = int(map)

	print(f'Stops after clustering: {len(stop_points)}')

	utils.json_utils.write_json_object(configs.STOP_REPLACEMENTS, final)
	utils.json_utils.write_json_object(configs.CLUSTERED_STOPS, stop_points)
	utils.json_utils.write_json_object(configs.CLUSTERED_ROUTES, routes)
	utils.json_utils.write_networkx_json(configs.FINAL_NETWORK, network)