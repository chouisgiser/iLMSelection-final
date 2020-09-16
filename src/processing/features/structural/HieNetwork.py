from igraph import *
import geopandas as gpd


# The door entity in the building
class DoorEntity:
    def __init__(self, name, geometry, floor, belong_edge):
        self.name = name
        self.geometry = geometry
        self.floor = floor
        self.belong_edge = belong_edge


# The room entity in the building
class RoomEntity:
    def __init__(self, name):
        self.name = name
        self.contain_doors = list()

    def add_door(self, door):
        if door in self.doordict:
            return
        self.contain_doors.append(door)


def has_point(ptlist, point):
    for index in range(len(ptlist)):
        if point.__eq__(ptlist[index]):
            return index
    return -1


class InNetwork(Graph):
    def __init__(self):
        Graph.__init__(self)
        self.vs["horizon_pos"] = []
        self.es["length"] = []

    def construct_network_by_shapefile(self,edge_file):
        edge_geometry = gpd.read_file(edge_file).geometry

        for index in range(len(edge_geometry)):
            line = edge_geometry[index]
            st_pt = Point(line.xy[0][0], line.xy[1][0])
            st_index = has_point(self.vs["horizon_pos"], st_pt)
            if st_index == -1:
                st_index = len(self.vs)
                self.add_vertex(st_index)
                self.vs[st_index]["horizon_pos"] = st_pt

            end_pt = Point(line.xy[0][len(line.xy[0])-1], line.xy[1][len(line.xy[1])-1])
            end_index = has_point(self.vs["horizon_pos"], end_pt)
            if end_index == -1:
                end_index = len(self.vs)
                self.add_vertex(end_index)
                self.vs[end_index]["horizon_pos"] = end_pt

            self.add_edge(st_index, end_index)
            self.es[len(self.es)-1]["length"] = line.length


network = InNetwork()
edge_file = "../data/taz_mm.tbl_routennetz_line.shp"
network.construct_network_by_shapefile(edge_file)
print(network.es["length"])
print(len(network.es))
