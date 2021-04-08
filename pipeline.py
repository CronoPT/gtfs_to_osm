'''
| This script just chains every script that is 
| involved in the process of preparing the road
| network to be used in the optimizaiton process
'''

import subprocess
import configs

if __name__ == '__main__':

	scripts = [
		# ['osm_network_to_json.py'], # get osm network
		# ['osm_manual_tweaks.py'],   # make small tweaks to the osm netwokr
		# ['network_to_geojson.py'],  # prepare network to be visualized
		# ['enhanced_projection.py'], # compute projections for gtfs stops
		# ['fix_projections.py'],     # fix one projection for each gtfs stop
		# ['stops_on_road.py'],       # make bus stops a part of the network
		# [			
		# 	'routes_to_geojson.py',
		# 	configs.ROUTES_STOP_SEQUENCE,
		# 	configs.NETWORK_WITH_STOPS,
		# 	configs.FINAL_MAPPINGS,
		# 	configs.ROUTE_SHAPES,
		# 	configs.STOP_LOCATIONS
		# ],  # make geojson files with the bus network before clustering
		['cluster_stops.py'],       # cluster stops together
		[
			'routes_to_geojson.py',
			configs.CLUSTERED_ROUTES,
			configs.FINAL_NETWORK,
			configs.CLUSTERED_STOPS,
			configs.CLUSTERED_ROUTE_SHAPES,
			configs.CLUSTERED_STOP_LOCATIONS
		],  # make geojson files with the bus network after clustering    
	]

	for index, script in enumerate(scripts):
		print(f'[PIPELINE] Starting {script[0]} - {index+1}/{len(scripts)}')
		out = subprocess.call(['python3', *script])
		if out != 0:
			exit()
		print(f'[PIPELINE] Finished {script[0]} - {index+1}/{len(scripts)}')
