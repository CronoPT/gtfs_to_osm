import pandas as pd
import json
import networkx as nx
import matplotlib.pyplot as plt

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

if __name__ == '__main__':
	stops_df = pd.read_csv('data/carris_gtfs/stops.txt', sep=',', decimal='.')
	route_df = pd.read_csv('data/carris_gtfs/shapes.txt', sep=',', decimal='.')

	file = open('data/lisbon_net_data.json', 'r')
	net  = json.load(file)
	file.close() 

	stop_ids = []
	not_stop = []
	for index, row in route_df.iterrows():
		print_progress_bar(index, route_df.shape[0], prefix='[STOP CHECK]   ')

		lat = row['shape_pt_lat']
		lon = row['shape_pt_lon']
		res = stops_df.loc[(stops_df['stop_lat']==lat) & (stops_df['stop_lon']==lon)]
		if res.shape[0] > 0:
			stop_ids.append(res.iloc[0]['stop_id'])
		else:
			stop_ids.append('Not Stop')
			not_stop.append([lon, lat])
	
	print_progress_bar(route_df.shape[0], route_df.shape[0], prefix='[STOP CHECK]   ')

	route_df['stop_id'] = stop_ids
	route_df = route_df[route_df['stop_id']!='Not Stop']

	nodes = {}
	graph_edges = []

	for index, shape in enumerate(route_df['shape_id'].unique()):
		print_progress_bar(index, len(route_df['shape_id'].unique()), prefix='[GRAPH BUILD]  ')

		sequence = route_df[ route_df['shape_id']==shape ]
		sequence.sort_values('shape_pt_sequence')

		prev = None
		for _, row in sequence.iterrows():
			lon = row['shape_pt_lon']
			lat = row['shape_pt_lat']
			stop_id = row['stop_id']
			coords  = (lon, lat)
			curr    = None 

			if coords not in nodes:
				curr = {
					'id':  len(nodes),
					'lon': lon,
					'lat': lat,
					'stop_id': stop_id
				}
				nodes[coords] = curr
			else:
				curr = nodes[coords]

			if prev != None:
				graph_edges.append((prev['id'], curr['id']))

			prev = curr
	
	print_progress_bar(len(route_df['shape_id'].unique()), 
	                   len(route_df['shape_id'].unique()), 
					   prefix='[GRAPH BUILD]  ')

	graph_nodes = []
	for _, obj in nodes.items():
		graph_nodes.append((obj['id'], {
			'lon': obj['lon'],
			'lat': obj['lat'],
			'stop_id': obj['stop_id']
		}))

	Gg = nx.Graph()
	Gg.add_nodes_from(graph_nodes)
	Gg.add_edges_from(graph_edges)

	file = open('data/stop_mappings.json', 'r')
	mappings = json.load(file)
	file.close() 

	p_nodes = {}
	p_graph_edges = []

	node_lon = nx.get_node_attributes(Gg, 'lon')
	node_lat = nx.get_node_attributes(Gg, 'lat')
	node_stop_ids = nx.get_node_attributes(Gg, 'stop_id')
	for index, edge in enumerate(Gg.edges):
		print_progress_bar(index, len(Gg.edges), prefix='[P GRAPH BUILD]')

		origin = edge[0]
		destin = edge[1]

		origin_stop_id = node_stop_ids[origin]
		destin_stop_id = node_stop_ids[destin]

		origin_map_item = list(filter(lambda item: item['stop_id']==origin_stop_id, mappings))[0]
		destin_map_item = list(filter(lambda item: item['stop_id']==destin_stop_id, mappings))[0]

		for origin_projection in origin_map_item['mappings']:
			for destin_projection in destin_map_item['mappings']:
				origin_point = origin_projection['point']
				destin_point = destin_projection['point']

				tuple_origin_point = origin_point[0], origin_point[1]
				tuple_destin_point = destin_point[0], destin_point[1]

				if tuple_origin_point not in p_nodes:
					p_nodes[tuple_origin_point] = len(p_nodes)

				if tuple_destin_point not in p_nodes:
					p_nodes[tuple_destin_point] = len(p_nodes)
				
				origin_node_id = p_nodes[tuple_origin_point]
				destin_node_id = p_nodes[tuple_destin_point]

				p_graph_edges.append((origin_node_id, destin_node_id))

	print_progress_bar(len(Gg.edges), len(Gg.edges), prefix='[P GRAPH BUILD]')

	p_graph_nodes = []
	for coords, node_id in p_nodes.items():
		p_graph_nodes.append((node_id, {
			'lon': coords[0],
			'lat': coords[1]
		}))
				
	Gp = nx.Graph()
	Gp.add_nodes_from(p_graph_nodes)
	Gp.add_edges_from(p_graph_edges)

	nx.draw_networkx(Gg, pos={
		node: (obj['lon'], obj['lat'])
		for node, obj in graph_nodes
	}, with_labels=False, node_size=10)

	plt.show()
	plt.clf()

	nx.draw_networkx(Gp, pos={
		node: (obj['lon'], obj['lat'])
		for node, obj in p_graph_nodes
	}, with_labels=False, node_size=10)

	plt.show()