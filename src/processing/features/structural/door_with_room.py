import geopandas as gpd
import os
import csv
import math
from shapely.geometry import Point, LineString
import numpy as np


def read_shapefile(file):
    rows = list()
    shape_data = gpd.read_file(file)
    for idx, row in shape_data.iterrows():
        # id = idx
        geometry = row["geometry"]
        newrow = [idx, geometry]
        rows.append(newrow)
    return rows


def network_construt(rooms,doors):
    room_node_ids = list()
    rooms_links = list()
    door_node_ids = list()
    room_door_links = list()
    for i in range(0,len(rooms)):
        polygon = rooms[i][1]
        # centroid = polygon.centroid
        # node = [i, rooms[i]["type"], centroid]
        if i not in room_node_ids:
            room_node_ids.append(i)
        for j in range(i+1,len(rooms)):
            next_polygon = rooms[j][1]
            if j not in room_node_ids:
                room_node_ids.append(j)
            if polygon.intersects(next_polygon) or polygon.touches(next_polygon):
                link = [i,j]
                rooms_links.append(link)

    for k in range(0, len(doors)):
        door_polygon = doors[k][1]
        if k not in door_node_ids:
            door_node_ids.append(k)
        for m in range(0, len(rooms)):
            newpolygon = rooms[m][1]
            if newpolygon.intersects(door_polygon) or newpolygon.touches(door_polygon):
                link = [m, k]
                room_door_links.append(link)


    print (len(rooms_links))
    print (len(room_door_links))
    return room_node_ids, door_node_ids, rooms_links, room_door_links


def write_network(rooms, doors, node_file, link_file, room_node_ids, door_node_ids, rooms_links, room_door_links):
    node_dataframe = gpd.GeoDataFrame()
    node_dataframe["geometry"] = None
    link_dataframe = gpd.GeoDataFrame()
    link_dataframe["geometry"] = None

    node_id = 0
    for room_node_id in room_node_ids:
        room_centroid = rooms[room_node_id][1].centroid
        node_dataframe.loc[node_id, "id"] = room_node_id
        node_dataframe.loc[node_id, "type"] = "room"
        node_dataframe.loc[node_id, "geometry"] = room_centroid
        node_id += 1
        # room_node = [room_node_id, "room", room_centroid]
        # room_nodes.append(room_node)

    for door_node_id in door_node_ids:
        door_centroid = doors[door_node_id][1].centroid
        node_dataframe.loc[node_id, "id"] = len(room_node_ids)+door_node_id
        node_dataframe.loc[node_id, "type"] = "door"
        node_dataframe.loc[node_id, "geometry"] = door_centroid
        node_id += 1
        # door_node = [len(room_nodes)+door_node_id, "door", door_centroid]
        # door_nodes.append(door_node)

    link_id = 0

    for rooms_link in rooms_links:
        pt1 = rooms[rooms_link[0]][1].centroid.coords[0]
        pt2 = rooms[rooms_link[1]][1].centroid.coords[0]
        line = LineString([pt1, pt2])
        link_dataframe.loc[link_id, "id"] = link_id
        link_dataframe.loc[link_id, "T1"] = rooms_link[0]
        link_dataframe.loc[link_id, "T2"] = rooms_link[1]
        link_dataframe.loc[link_id, "geometry"] = line
        # link = [link_num, rooms_link[0], rooms_link[1], line]
        link_id += 1
        # links.append(link)

    for room_door_link in room_door_links:
        pt1 = rooms[room_door_link[0]][1].centroid.coords[0]
        pt2 = doors[room_door_link[1]][1].centroid.coords[0]
        line = LineString([pt1, pt2])
        link_dataframe.loc[link_id, "id"] = link_id
        link_dataframe.loc[link_id, "T1"] = room_door_link[0]
        link_dataframe.loc[link_id, "T2"] = room_door_link[1] + len(rooms)
        link_dataframe.loc[link_id, "geometry"] = line
        # link = [link_num, room_door_link[0], room_door_link[1], line]
        link_id += 1
        # links.append(link)

    print(node_id)
    print(link_id)

    node_dataframe.to_file(node_file)
    link_dataframe.to_file(link_file)


room_file = "../data/Y25/Floor_L_room.shp"
door_file = "../data/Y25/buffer/Floor_L_door_buffer.shp"

rooms = read_shapefile(room_file)
doors = read_shapefile(door_file)

room_node_ids, door_node_ids, rooms_links, room_door_links = network_construt(rooms,doors)

out_node = "data/Y25_Network/Floor_L_node.shp"
out_link = "data/Y25_Network/Floor_L_link.shp"
write_network(rooms, doors, out_node, out_link, room_node_ids, door_node_ids, rooms_links, room_door_links)
