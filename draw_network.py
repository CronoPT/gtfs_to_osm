import networkx as nx
import matplotlib.pyplot as plt
import utils.json_utils
import configs

if __name__ == '__main__':

	lisbon_net = utils.json_utils.read_network_json(configs.TWEAKED_NETWORK)

	plt.figure(figsize=(20, 20))
	nx.draw_networkx(lisbon_net, pos={
		node: (info['x'], info['y'])
		for node, info in lisbon_net.nodes(data=True)
	}, with_labels=False, node_size=0.5, arrowsize=2, width=0)

	shapes = utils.json_utils.read_json_object(configs.NETWORK_LINKS)
	for line in shapes['coordinates']:
		prev_point = None
		for curr_point in line:
			if prev_point != None: 
				plt.plot([curr_point[0], prev_point[0]], [curr_point[1], prev_point[1]], linewidth=0.2, color='black')
			prev_point = curr_point

	plt.savefig('lisbon_net.png', dpi=500)