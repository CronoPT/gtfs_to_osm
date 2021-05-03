'''
| The value within which we will look for projected 
| stops on the road network.
'''
THRESHOLD = 15

'''
| This script indexes road segments in a greed to make
| computation faster, this is the number of divisions
| to make in one dimension, meaning that the whole
| road will be divided into DIVISIONS^2 squares.
'''
DIVISIONS = 100

'''
| The name of the file where the network
| retrieved from Open Street Map will be
| stored.
'''
OSM_NETWORK = 'data/json/osm_network.json'

'''
| Some manual adjustments might be needed,
| this file will store the road network
| retrieved from Open Street Map after the
| tweaks are introduced.
'''
TWEAKED_NETWORK = 'data/json/tweaked_network.json'

'''
| QGIS is being used as a tool to visualize
| geographic structers. This file will save 
| the shape of the road network so it can 
| be visualized.
'''
NETWORK_LINKS = 'data/geojson/network_links.geojson'

'''
| QGIS is being used as a tool to visualize
| geographic structers. This file will save 
| the node locations of the road network so 
| it can be visualized.
'''
NETWORK_NODES = 'data/geojson/network_nodes.geojson'

'''
| Several projections of a GTFS stop location
| compete for a place in the final network, 
| this file saves, for every GTFS stop, its
| candidates.
'''
CANDIDATE_MAPPINGS = 'data/json/candidate_mappings.json'

'''
| Every GTFS stop has a final projection in the
| road network, this files saves them.
'''
FINAL_MAPPINGS = 'data/json/final_mappings.json'

'''
| This file saves a sequence of stop identifiers
| for each route.
'''
ROUTES_STOP_SEQUENCE = 'data/json/routes_stop_sequence.json'

'''
| QGIS is being used as a tool to visualize
| geographic structers. This file will save 
| the shape of all the routes so we can then
| visualize them.
'''
ROUTE_SHAPES = 'data/geojson/route_shapes.geojson'

'''
| QGIS is being used as a tool to visualize
| geographic structers. This file will save 
| a point for every bus stop location so we 
| can then visualize them.
'''
STOP_LOCATIONS = 'data/geojson/stop_locations.geojson'

'''
| The network after having all the stop 
| fixed projections as nodes in the network
'''
NETWORK_WITH_STOPS = 'data/json/network_with_stops.json'

FINAL_NETWORK = 'data/json/final_network.json'

CLUSTERED_STOPS = 'data/json/clustered_stop_locations.json'

CLUSTERED_ROUTES = 'data/json/clustered_routes_stop_sequence.json'

CLUSTERED_STOP_LOCATIONS = 'data/geojson/clustered_stop_locations.geojson'

CLUSTERED_ROUTE_SHAPES = 'data/geojson/clustered_route_shapes.geojson'

STOP_REPLACEMENTS = 'data/json/stop_replacements.json'