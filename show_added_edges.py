
import networkx as nx
import osmnx as ox
import math
import utils.json_utils
import utils.geometric_utils
import configs
import geojsonio
import json

if __name__ == '__main__':

	road_net = utils.json_utils.read_network_json(configs.TWEAKED_NETWORK)

	new_pairs = [{
		'origin': [-9.174212, 38.677772],
		'destin': [-9.174522, 38.678471]
	},{
		'origin': [-9.174522, 38.678471],
		'destin': [-9.174802, 38.678989]
	},{
		'origin': [-9.176340, 38.717388],
		'destin': [-9.169120, 38.722614],
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

	lines = []

	for obj in new_pairs:
		origin = [obj['origin'][1], obj['origin'][0]]
		destin = [obj['destin'][1], obj['destin'][0]]
		origin_node_id = int(ox.distance.get_nearest_node(road_net, origin))
		destin_node_id = int(ox.distance.get_nearest_node(road_net, destin))

		edge_data = road_net.get_edge_data(origin_node_id, destin_node_id)
		
		if edge_data == None:
			continue

		geo = []
		if 'geometry' in edge_data[0]:
			geo = edge_data[0]['geometry']
		else:
			geo = [
				[float(road_net.nodes[origin_node_id]['x']), float(road_net.nodes[origin_node_id]['y'])],
				[float(road_net.nodes[destin_node_id]['x']), float(road_net.nodes[destin_node_id]['y'])]
			]
		lines.append(geo)

	geojson_object = {
		"type": "MultiLineString",
		"coordinates": lines
	}

	print(json.dumps(geojson_object, indent=4))

	geojsonio.display(json.dumps(geojson_object, indent=4))