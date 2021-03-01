import networkx as nx
import matplotlib.pyplot as plt
import json

if __name__ == '__main__':
	file = open('data/network.json', 'r')
	network_json = json.load(file)
	file.close() 	

	lisbon_net = nx.readwrite.json_graph.adjacency_graph(network_json)
	plt.figure(figsize=(20, 20))
	nx.draw_networkx(lisbon_net, pos={
		node: (info['x'], info['y'])
		for node, info in lisbon_net.nodes(data=True)
	}, with_labels=False, node_size=0.5, arrowsize=2, width=0)

	file = open('data/lisbon_net_links.geojson', 'r')
	shapes = json.load(file)
	file.close() 	

	for line in shapes['coordinates']:
		prev_point = None
		for curr_point in line:
			if prev_point != None: 
				plt.plot([curr_point[0], prev_point[0]], [curr_point[1], prev_point[1]], linewidth=0.2, color='black')
			prev_point = curr_point

	plt.savefig('lisbon_net.png', dpi=500)