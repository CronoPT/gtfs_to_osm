'''
| This script will mainly be used so i eventually import the Lisbon road network
| into a C++ structure. It grabs the network from Open Stree Map with the osmnx python
| package and then produces a json file which is just a list of nodes and a list
| of adjancencies.
'''

import osmnx as ox
import networkx as nx
import json

'''
| This will fetch the map and put into a NetworkX
| MultiDiGraph.
'''
lisbon_net = ox.graph.graph_from_address(
    'lisbon', 
    dist=6000,
    retain_all=False,
    network_type='drive',
    simplify=True,
    dist_type='bbox'
)

'''
| This structure will look something like this:
|{
|    'nodes': [
|        {
|            'x':  ..., 
|            'y':  ...,
|            'id': ...,
|        },
|        ...,
|        {
|            'x':  ..., 
|            'y':  ...,
|            'id': ...,
|        }
|    ]
|    'adjacency': [
|        [
|            {
|                ..., 
|                'lenght': ...,
|                'geometry': [[<x>, <y>], ...,  [<x>, <y>]],
|                'id': ...,
|                ...
|            },
|            ...,
|            {
|                ..., 
|                'lenght': ...,
|                'geometry': [[<x>, <y>], ...,  [<x>, <y>]],
|                'id': ...,
|                ...
|            }
|        ],
|        ...,
|        [...]
|    ]
|}
|
| So basically, there's a list of nodes, each with its own coordinates and then
| there's a list of adjancencies. Theres a list of adjancencies for each node.
| Each adjancency's 'id' attribute is the id of the node connecting to the 
| concerned node. If 'geometry' is omitted, then there is no complicated 
| road shape connecting the nodes, the road is a straight line.
'''
json_data  = nx.readwrite.json_graph.adjacency_data(lisbon_net)

json_data.pop('directed')
json_data.pop('multigraph')
json_data.pop('graph')

'''
| The links are represented as a shapely Multiline, I'm
| just turning them into a list of coordinates so i can 
| read this file into C++ if need be.
'''
for adjs in json_data['adjacency']:
	for adj in adjs:
		for key, value in adj.items():
			if key=='geometry':
				adj['geometry'] = [(x, y) for x, y in value.coords]

with open('data/lisbon_net_data.json', 'w') as json_file:
	json.dump(json_data, json_file, indent=4)