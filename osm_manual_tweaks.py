
import networkx as nx
import json
import osmnx as ox
import math


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


def add_new_edges(network, edges):
	for edge in edges:
		origin = edge['origin']
		destin = edge['destin']
		ox_origin = (origin[1], origin[0])
		ox_destin = (destin[1], destin[0])

		origin_node_id = int(ox.distance.get_nearest_node(network, ox_origin))
		destin_node_id = int(ox.distance.get_nearest_node(network, ox_destin))

		origin_node = network.nodes[origin_node_id]
		destin_node = network.nodes[destin_node_id]

		origin_point = [origin_node['x'], origin_node['y']] 
		destin_point = [destin_node['x'], destin_node['y']]

		attrs = {
			'id': destin_node_id,
			'key': 0,
			'manual': True 
		}

		length = 0
		if 'geometry' in edge:
			edge['geometry']  = [origin_point] + edge['geometry'] + [destin_point]
			attrs['geometry'] = edge['geometry']
			
			prev_point = None
			for curr_point in edge['geometry']:
				if prev_point != None:
					length += haversine_distance(prev_point, curr_point)*1000
				prev_point = curr_point	

		else:
			length = haversine_distance(origin_point, destin_point)*1000

		attrs['length'] = length

		network.add_edge(origin_node_id, destin_node_id, **attrs)


def add_inv_edges(network, edges):
	for edge in edges:
		origin = edge['origin']
		destin = edge['destin']
		ox_origin = (origin[1], origin[0])
		ox_destin = (destin[1], destin[0])

		origin_node_id = int(ox.distance.get_nearest_node(network, ox_origin))
		destin_node_id = int(ox.distance.get_nearest_node(network, ox_destin))

		og_edge = network.get_edge_data(origin_node_id, destin_node_id, key=edge['key'])

		if 'key' in og_edge:
			og_edge.pop('key') 

		network.add_edge(destin_node_id, origin_node_id, key=edge['key'], **og_edge)
		if 'geometry' in og_edge:
			coord_list = [coord for coord in og_edge['geometry']]
			coord_list.reverse()
			network.add_edge(
				destin_node_id, 
				origin_node_id, 
				edge['key'], 
				geometry=coord_list,
				manual=True
			)


if __name__ == '__main__':
	file = open('data/osm_network.json', 'r')
	network_json = json.load(file)
	file.close() 

	road_net = nx.readwrite.json_graph.adjacency_graph(network_json)

	new_pairs = [{
		'origin': [-9.174212, 38.677772],
		'destin': [38.678471, -9.174522]
	},{
		'origin': [-9.174522, 38.678471],
		'destin': [-9.174802, 38.678989]
	},{
		'origin': [-9.176340, 38.717388],
		'destin': [-9.169120, 38.722614],
		'geometry': [
			[-9.17595, 38.71848],
			[-9.17583, 38.71889],
			[-9.17574, 38.71904],
			[-9.17568, 38.71914],
			[-9.17563, 38.71919],
			[-9.17549, 38.71931],
			[-9.175,   38.71962],
			[-9.17393, 38.72035],
			[-9.1738,  38.72046],
			[-9.17365, 38.7206 ],
			[-9.17356, 38.72068],
			[-9.17341, 38.72082],
			[-9.17272, 38.72151],
			[-9.1726,  38.7216 ],
			[-9.1725,  38.72168],
			[-9.17237, 38.72176],
			[-9.17211, 38.72187],
			[-9.1718,  38.72199]
		]
	}]

	inv_pairs = [{
		'origin': [-9.177902, 38.708993],
		'destin': [-9.174802, 38.678989],
		'key': 0
	},{
		'origin': [-9.177867, 38.709833],
		'destin': [-9.177870, 38.709589],
		'key': 0
	},{
		'origin': [-9.177870, 38.709589],
		'destin': [-9.177917, 38.708967],
		'key': 0
	}]

	add_new_edges(road_net, new_pairs)
	add_inv_edges(road_net, inv_pairs)

	json_data  = nx.readwrite.json_graph.adjacency_data(road_net)
	with open('data/network.json', 'w') as json_file:
		json.dump(json_data, json_file, indent=4)