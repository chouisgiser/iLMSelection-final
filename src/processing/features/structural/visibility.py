from utils import file_io
from shapely.geometry import Point, LineString, Polygon


def get_visibility(floor, visible_threshold):
    visibility_list = list()
    pois_file = '../data/Y25/POI/Floor_' + floor + '_POI.shp'
    corridor_label_file = '../data/Y25/grid/Floor_' + floor + '_corridor_label.shp'
    corridor_file = '../data/Y25/corridor/Floor_' + floor + '_corridor.shp'
    pois = file_io.read_shapefile(pois_file, ['name'])
    corridor_labels = file_io.read_shapefile(corridor_label_file, [])
    corridor = file_io.read_shapefile(corridor_file, [])
    for poi in pois:
        label_count = 0
        for corridor_label in corridor_labels:
            line = LineString([poi[1].coords[0], corridor_label[1].coords[0]])
            length = line.length / 100
            if length < visible_threshold:
                if line.within(corridor[0][1]):
                    label_count += 1
        poi_dict = {'name': poi[2], 'visibility': label_count}
        visibility_list.append(poi_dict)

    return visibility_list


floors = ['H', 'J', 'K', 'L']
visibility_list = list()
for floor in floors:
    visibility_floor = get_visibility(floor, 20)
    visibility_list.extend(visibility_floor)

headers = ["name", "visibility"]
file_io.write_csv_file('visibility.csv', visibility_list, headers)
