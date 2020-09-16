import geopandas as gpd
from igraph import *
from random import randint
import os
import csv
import math
from shapely.geometry import Point, LineString
import numpy as np


class BuildingGraph(Graph):
    def __init__(self, building_id):
        self.id = building_id
        Graph.__init__(self)
        self.vs["vid_in_floor"] = []
        self.vs["floorid"] = []
        self.vs["geometry"] = []
        self.vs["type"] = []
        self.vs["level"] = []
        self.vs["function"] = []
        self.vs["number"] = []

        self.es["eid_in_floor"] = []
        self.es["floorid"] = []
        self.es["geometry"] = []
        self.es["level"] = []
        self.es["length"] = []

        # self.nodes = list()
        # self.links = list()

    def add_floor(self, floor):
        num = len(self.vs)
        for node in floor.nodes:
            # veterx_index = len(self.vs)
            self.add_vertex()
            # self.vs[veterx_index]["floorid"] = floor.id
            # self.vs[veterx_index]["vid_in_floor"] = node[0]
            # self.vs[veterx_index]["geometry"] = node[1]
            # self.vs[veterx_index]["type"] = node[2]
            # self.vs[veterx_index]["level"] = node[3]
            # self.vs[veterx_index]["function"] = node[4]
            # self.vs[veterx_index]["number"] = node[5]
            self.vs[-1]["floorid"] = floor.id
            self.vs[-1]["vid_in_floor"] = node[0]
            self.vs[-1]["geometry"] = node[1]
            self.vs[-1]["type"] = node[2]
            self.vs[-1]["level"] = node[3]
            self.vs[-1]["function"] = node[4]
            self.vs[-1]["number"] = node[5]

        for edge in floor.links:
            # edge_index = len(self.es)
            id1 = int(num + edge[2])
            id2 = int(num + edge[3])
            self.add_edge(id1, id2)
            # self.es[edge_index]["floorid"] = floor.id
            # self.es[edge_index]["eid_in_floor"] = edge[0]
            # self.es[edge_index]["geometry"] = edge[1]
            # self.es[edge_index]["level"] = edge[4]
            # self.es[edge_index]["length"] = edge[1].length
            self.es[-1]["floorid"] = floor.id
            self.es[-1]["eid_in_floor"] = edge[0]
            self.es[-1]["geometry"] = edge[1]
            self.es[-1]["level"] = edge[4]
            self.es[-1]["length"] = edge[1].length/100
            # print(self.es[-1]["length"])

    def add_links_between_floors(self, transition_pair):
        vertex_id_1 = transition_pair["vertex_ids"][0]
        vertex_id_2 = transition_pair["vertex_ids"][1]

        self.add_edge(vertex_id_1, vertex_id_2)
        self.es[- 1]["level"] = 1.5
        if self.vs[vertex_id_1]["function"] == "lift":
            self.es[- 1]["length"] = 8
        if self.vs[vertex_id_1]["function"] == "staircase":
            self.es[- 1]["length"] = 16

    def get_id_by_number(self, number):
        id_list = list()
        for vertex in self.vs:
            if vertex["number"] == number:
                id_list.append(vertex.index)
        return id_list

    def get_room_closeness(self, vids):
        rm_closeness = 0
        for vid in vids:
            rm_closeness += self.closeness(vid, weights="length")
        rm_closeness = rm_closeness/(len(vids))

        return rm_closeness

    def get_room_betweenness(self, vids):
        rm_betweenness = 0
        for vid in vids:
            rm_betweenness += self.betweenness(vid, weights="length")
        rm_betweenness = rm_betweenness / (len(vids))

        return rm_betweenness

    def get_room_betweenness_by_door(self, vids):
        rm_betweenness = 0
        rm_neighbors = list()
        for vid in vids:
            tmp_neighbors = self.neighbors(vid)
            for neighbor in tmp_neighbors:
                if self.vs[neighbor]["type"] != "door":
                    rm_neighbors

        for rm_neighbor in rm_neighbors:
            rm_betweenness += self.betweenness(rm_neighbor)
        rm_betweenness = rm_betweenness / (len(rm_neighbors))

        return  rm_betweenness

    def get_room_control(self, vids, number):
        rm_control = 0
        rm_neighbors = set()
        for vid in vids:
            tmp_neighbors = self.neighbors(vid)
            for neighbor in tmp_neighbors:
                if self.vs[neighbor]["number"] != number and self.vs[neighbor]["type"] != number:
                    rm_neighbors.add(neighbor)

        for rm_neighbor in rm_neighbors:
            print(self.degree(rm_neighbor))
            rm_control += 1/self.degree(rm_neighbor)

        return rm_control

    def get_proximity_2_dp(self, vids):
        dp_ids = self.vs.select(function="junction")
        shortest_paths_dis = list()
        for vid in vids:
            neighbor_ids = self.neighbors(vid)
            for neighbor_id in neighbor_ids:
                if self.vs[neighbor_id]["type"] == "door":
                    tmp_paths_dis = self.shortest_paths_dijkstra(neighbor_id, dp_ids, "length")
                    shortest_paths_dis.append(min(tmp_paths_dis[0]))

        shortest_dis = min(shortest_paths_dis)
        if shortest_dis == 0:
            proximity = 1
        else:
            proximity = 1/shortest_dis

        return proximity

    def get_proximity_2_fe(self, vids):
        dp_ids = self.vs.select(function_in=["staircase", "lift"])
        shortest_paths_dis = list()
        for vid in vids:
            neighbor_ids = self.neighbors(vid)
            for neighbor_id in neighbor_ids:
                if self.vs[neighbor_id]["type"] == "door":
                    tmp_paths_dis = self.shortest_paths_dijkstra(neighbor_id, dp_ids, "length")
                    shortest_paths_dis.append(min(tmp_paths_dis[0]))

        shortest_dis = min(shortest_paths_dis)
        if shortest_dis == 0:
            proximity = 1
        else:
            proximity = 1/shortest_dis

        return proximity

    def get_proximity_2_be(self, vids):
        dp_ids = self.vs.select(function="entrance")
        shortest_paths_dis = list()
        for vid in vids:
            neighbor_ids = self.neighbors(vid)
            for neighbor_id in neighbor_ids:
                if self.vs[neighbor_id]["type"] == "door":
                    tmp_paths_dis = self.shortest_paths_dijkstra(neighbor_id, dp_ids, "length")
                    shortest_paths_dis.append(min(tmp_paths_dis[0]))

        shortest_dis = min(shortest_paths_dis)
        if shortest_dis == 0:
            proximity = 1
        else:
            proximity = 1/shortest_dis

        return proximity


class Floor:

    def __init__(self, floor_id):
        self.id = floor_id
        self.nodes = list()
        self.links = list()

    def read_node_shp(self, directory):
        # noderows = list()
        floor_node_file = directory + "/" + "Floor_" + self.id + "_node.shp"
        shape_data = gpd.read_file(floor_node_file)
        for idx, row in shape_data.iterrows():
            # id = idx
            geometry = row["geometry"]
            node_type = row[1]
            level = row["level"]
            function = row["function"]
            number = row["number"]
            newrow = [idx, geometry, node_type, level, function, number]
            self.nodes.append(newrow)

    def read_link_shp(self, directory):
        # linkrows = list()
        floor_link_file = directory + "/" + "Floor_" + self.id + "_link.shp"
        shape_data = gpd.read_file(floor_link_file)
        for idx, row in shape_data.iterrows():
            # id = idx
            geometry = row["geometry"]
            T1 = row["T1"]
            T2 = row["T2"]
            level = row["level"]
            id_infloor = idx
            newrow = [idx, geometry, T1, T2, level]
            self.links.append(newrow)
        # return linkrows
