
import networkx as nx
import osmnx as ox
import math
import utils.json_utils
import utils.geometric_utils
import configs

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
					length += utils.geometric_utils.haversine_distance(prev_point, curr_point)
				prev_point = curr_point	

		else:
			length = utils.geometric_utils.haversine_distance(origin_point, destin_point)

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

	road_net = utils.json_utils.read_network_json(configs.OSM_NETWORK)

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

	utils.json_utils.write_networkx_json(configs.TWEAKED_NETWORK, road_net)
