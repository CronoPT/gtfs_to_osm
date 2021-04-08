import sys
import utils.json_utils
import networkx as nx
import numpy as np


if __name__ == '__main__':
	route_paths = utils.json_utils.read_json_object(sys.argv[1])
	road_net    = utils.json_utils.read_network_json(sys.argv[2])
	stop_points = utils.json_utils.read_json_object(sys.argv[3])

	routes = []
	for route_info in route_paths:
		stops  = route_info['stops']
		origin = None
		route  = []
		for index, destin in enumerate(stops):
			if origin != None:
				path = nx.algorithms.shortest_paths.weighted.dijkstra_path(
					road_net, 
					source=origin, 
					target=destin,
					weight='length'
				)

				path_in_coordinates = []
				origin_node = None
				for index, destin_node in enumerate(path):
					if origin_node != None:
						edge_info = road_net.get_edge_data(origin_node, destin_node)

						best_edge = None
						best_cost = np.inf
						for key, edge_data in edge_info.items():
							length = edge_data['length']
							if length < best_cost:
								best_cost = length
								best_edge = edge_data 

						if 'geometry' in best_edge:
							path_in_coordinates.extend(best_edge['geometry'])
						else:
							path_in_coordinates.extend([
								[road_net.nodes[origin_node]['x'], road_net.nodes[origin_node]['y']],
								[road_net.nodes[destin_node]['x'], road_net.nodes[destin_node]['y']]
							])
					origin_node = destin_node

				if index==0:
					route.extend(path_in_coordinates)
				else:
					route.extend(path_in_coordinates[1:])

			origin = destin
		
		routes.append(route)

	utils.json_utils.write_geojson_lines(
		sys.argv[4], 
		routes
	)
	utils.json_utils.write_geojson_points(
		sys.argv[5],
		[item['point'] for item in stop_points]
	)

