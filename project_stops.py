'''
| This script generates will project the bus stops 
| present in the GTFS specification to actual points
| on the road network using the algorithm described
| by Vuurstaek et al. 
'''

import pandas as pd
import numpy as np
import json
import math

def print_progress_bar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    '''
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    '''

    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    if iteration == total: 
        print()

'''
| The value within which we will look for projected 
| stops on the road network
'''
THRESHOLD = 15

def closest_point(a, b, p):
    '''
    | This function finds the point outside ab 
    | that is closest to point p.
    '''

    a = np.array(a)
    b = np.array(b)
    p = np.array(p)
    t = - np.inner((b-a), (a-p)) / np.inner((b-a), (b-a))

    if t > 0 and t < 1:
        return ((1-t)*a + t*b).tolist() 

    g0 = np.linalg.norm((a - p))
    g1 = np.linalg.norm((b - p))

    return a.tolist() if g0 < g1 else b.tolist()

def haversine_distance(a, b):
    '''
    | This function computes the haversine distance
    | between point a and point b. The distance is
    | in kilometers.
    '''

    lat1 = a[1]
    lat2 = b[1]
    lon1 = a[0]
    lon2 = b[0]
    lat1, lat2, lon1, lon2 = map(math.radians, [lat1, lat2, lon1, lon2])

    d_lat = lat2 - lat1
    d_lon = lon2 - lon1

    temp = (
          math.sin(d_lat/2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(d_lon/2) ** 2
    )

    return 6373.0 * (2 * math.atan2(math.sqrt(temp), math.sqrt(1 - temp)))


stops_df = pd.read_csv('data/carris_gtfs/stops.txt', sep=',', decimal='.')
route_df = pd.read_csv('data/carris_gtfs/shapes.txt', sep=',', decimal='.')

file = open('data/lisbon_net_data.json', 'r')
net  = json.load(file)
file.close() 

stop_ids = []
not_stop = []
for index, row in route_df.iterrows():
    lat = row['shape_pt_lat']
    lon = row['shape_pt_lon']
    res = stops_df.loc[(stops_df['stop_lat']==lat) & (stops_df['stop_lon']==lon)]
    if res.shape[0] > 0:
        stop_ids.append(res.iloc[0]['stop_id'])
    else:
        stop_ids.append('Not Stop')
        not_stop.append([lon, lat])

route_df['route_id'] = stop_ids
route_df = route_df[route_df['route_id']!='Not Stop']

stop_mappings = []
for index, row in stops_df.iterrows():

    print_progress_bar(index, stops_df.shape[0])

    lat = row['stop_lat']
    lon = row['stop_lon']
    stp = [lon, lat]
    pjs = [] 
    for index, adjacencies in enumerate(net['adjacency']):
        for link in adjacencies:
            if 'geometry' in link:
                origin = None
                destin = None
                for coords in link['geometry']:
                    destin = coords
                    if origin != None:
                        closest  = closest_point(origin, destin, stp)
                        distance = haversine_distance(stp, closest)*1000
                        if distance < THRESHOLD:
                            pjs.append({
                                'point': closest,
                                'distance': distance
                            })
                    origin = destin

            else:
                destin_item = list(filter(lambda item: item['id']==link['id'], net['nodes']))[0]
                origin = [net['nodes'][index]['x'], net['nodes'][index]['y']]
                destin = [destin_item['x'], destin_item['y']]
                closest  = closest_point(origin, destin, stp)
                distance = haversine_distance(stp, closest)*1000
                if distance < THRESHOLD:
                    pjs.append({
                        'point': closest,
                        'distance': distance
                    })

    stop_mappings.append({
        'stop_id':  row['stop_id'],
        'point':    stp,
        'mappings': pjs
    })

print_progress_bar(stops_df.shape[0], stops_df.shape[0])

json_data = {
    'type': 'MultiPoint',
    'coordinates': not_stop
}

with open('data/not_stops_gtfs.geojson', 'w') as json_file:
	json.dump(json_data, json_file, indent=4)

with open('data/stop_mappings.json', 'w') as json_file:
    json.dump(stop_mappings, json_file, indent=4)