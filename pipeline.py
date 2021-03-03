import subprocess

if __name__ == '__main__':
	scripts = [
		# 'osm_network_to_json.py', 
		# 'osm_manual_tweaks.py',
		# 'network_to_geojson.py',
		'enhanced_projection.py',
		'fix_projections.py'
	]

	for index, script in enumerate(scripts):
		print('[PIPELINE] Starting {} - {}/{}'.format(
			script, index+1, len(scripts)
		))
		subprocess.call(['python3', script])
		print('[PIPELINE] Finished {} - {}/{}'.format(
			script, index+1, len(scripts)
		))