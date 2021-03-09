'''
| This script just chains every script that is 
| involved in the process of preparing the road
| network to be used in the optimizaiton process
'''

import subprocess

if __name__ == '__main__':

	scripts = [
		'osm_network_to_json.py', # get osm network
		'osm_manual_tweaks.py',   # make small tweaks to the osm netwokr
		'network_to_geojson.py',  # prepare network to be visualized
		'enhanced_projection.py', # compute projections for gtfs stops
		'fix_projections.py',     # fix one projection for each gtfs stop
		'stops_on_road.py'        # make bus stops a part of the network
	]

	for index, script in enumerate(scripts):
		print(f'[PIPELINE] Starting {script} - {index+1}/{len(scripts)}')
		subprocess.call(['python3', script])
		print(f'[PIPELINE] Finished {script} - {index+1}/{len(scripts)}')