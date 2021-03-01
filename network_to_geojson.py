'''
| This script will mainly be used so i can visualize the Lisbon road network
| in QGIS. It grabs the network from Open Stree Map with the osmnx python
| package and then produces two .geoson files, one with all the road junctions
| (or nodes) in the city of Lisbon and other with all the connections (or links).
'''

import osmnx as ox
import networkx as nx
import json


if __name__ == '__main__':
	
	file = open('data/network.json', 'r')
	json_data = json.load(file)
	file.close()

	nodes_list = []
	links_list = []

	for node in json_data['nodes']:
		nodes_list.append([node['x'], node['y']])

	'''
	| The links are represented as a shapely Multiline, I'm
	| just turning them into a list of coordinates so it all
	| is in order with the geojson standart.
	'''
	for index, adjacencies in enumerate(json_data['adjacency']):
		for link in adjacencies:
			if 'geometry' in link:
				links_list.append([[x, y] for x, y in link['geometry']])
			else:
				'''
				| Whenever a road segment has no curves, i.e. it is only
				| a line, no coordinates are used, instead it is implied
				| that we are connecting the origin node with the node
				| identified by 'id'. This else statment handles that.
				'''
				destiny_item = list(filter(lambda item: item['id']==link['id'], json_data['nodes']))[0]
				origin  = [json_data['nodes'][index]['x'], json_data['nodes'][index]['y']]
				destiny = [destiny_item['x'], destiny_item['y']]
				links_list.append([origin, destiny])

	nodes = {
		'type': 'MultiPoint',
		'coordinates': nodes_list
	}

	links = {
		'type': 'MultiLineString',
		'coordinates': links_list
	}

	with open('data/lisbon_net_nodes.geojson', 'w') as json_file:
		json.dump(nodes, json_file, indent=4)

	with open('data/lisbon_net_links.geojson', 'w') as json_file:
		json.dump(links, json_file, indent=4)