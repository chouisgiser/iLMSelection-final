import geopandas as gpd
from igraph import *
from random import randint
import os
import csv
import math
from shapely.geometry import Point, LineString
import numpy as np


def _plotbycate(g):
    vs_outdegree = g.outdegree()
    vs_betweenness = g.betweenness()
    vs_closeness = g.closeness()

    visual_style = {}

    # Chooose the layout
    layout = g.layout("reingold_tilford_circular")
    visual_style["layout"] = layout

    # Set bbox and margin
    visual_style["bbox"] = (3072, 2048)
    visual_style["margin"] = 150

    # Set vertices size
    visual_style["vertex_size"] = [w / max(vs_closeness) * 18 + 8 for w in vs_closeness]

    # Set vertices lables
    #     visual_style["vertex_label"] = [str(vs_name) + '\n' + str(w_name) for vs_name, w_name in
    #                                     zip(g.vs["number"], vs_closeness)]
    visual_style["vertex_label"] = [str(vs_name) for vs_name in zip(g.vs["function"])]

    color_dict = {1.0: "red", 2.0: "pink", 3.0: "light blue", 4.0: "light green"}
    visual_style["vertex_color"] = [color_dict[category] for category in g.vs["level"]]

    # Set label size
    visual_style["vertex_label_size"] = 10

    # Set label position
    visual_style["vertex_label_dist"] = 1.2

    plot(g,"type.png", **visual_style)

# plot graph based on the community detection algorithms
def _plotbycd(g, membership = None):
    vs_outdegree = g.outdegree()
    vs_betweenness = g.betweenness()
    vs_closeness = g.closeness()

    visual_style = {}

    if membership is not None:
        gcopy = g.copy()
        edges = []
        edges_colors = []
        for edge in g.es():
            if membership[edge.tuple[0]] != membership[edge.tuple[1]]:
                edges.append(edge)
                edges_colors.append("gray")
            else:
                edges_colors.append("black")
        gcopy.delete_edges(edges)
        layout = gcopy.layout("kk")
        visual_style["edge_color"] = edges_colors
        # g.es["color"] = edges_colors
    else:
        layout = g.layout("kk")
        visual_style["edge_color"] = "gray"
        # g.es["color"] = "gray"

    # Chooose the layout
    visual_style["layout"] = layout

    # Set bbox and margin
    visual_style["bbox"] = (3072, 2048)
    visual_style["margin"] = 150

    # Set vertices size
    visual_style["vertex_size"] = [w / max(vs_closeness) * 18 + 8 for w in vs_closeness]

    # Set vertices lables
    # visual_style["vertex_label"] = [str(vs_name) + '\n' + str(w_name) for vs_name, w_name in
    #                                 zip(g.vs["number"], vs_closeness)]
    visual_style["vertex_label"] = [str(vs_name) for vs_name in zip(g.vs["level"])]

    if membership is not None:
        # Define colors used for outdegree visualization
        colors = []
        for i in range(0, max(membership) + 1):
            colors.append(str("#") + '%06X' % randint(0, 0xFFFFFF))
        visual_style["vertex_color"] = [colors[membership[vertex.index]] for vertex in g.vs()]

    # Set label size
    visual_style["vertex_label_size"] = 15

    # Set label position
    visual_style["vertex_label_dist"] = 1.2


    plot(g,"community.png", **visual_style)


def _plotbycentrality(g):
    vs_outdegree = g.outdegree()
    vs_betweenness = g.betweenness()
    vs_closeness = g.closeness()

    visual_style = {}

    # Chooose the layout
    layout = g.layout("kk")
    visual_style["layout"] = layout

    # Set bbox and margin
    visual_style["bbox"] = (3072, 2048)
    visual_style["margin"] = 150

    # Set vertices size
    # visual_style["vertex_size"] = [w / max(vs_betweenness) * 25 + 50 for w in vs_betweenness]
    #
    # # Set vertices lables
    # visual_style["vertex_label"] = [str(vs_name) + '\n' + str(w_name) for vs_name, w_name in
    #                                 zip(g.vs["name"], vs_betweenness)]

    # Set vertices size
    visual_style["vertex_size"] = [w / max(vs_closeness) * 8 + 5 for w in vs_closeness]

    # Set vertices lables
    # visual_style["vertex_label"] = [str(vs_name) + '\n' + str(w_name) for vs_name, w_name in
    #                                 zip(g.vs["number"], vs_closeness)]

    # colors = []
    # for i in range(0, max(vs_outdegree)+1,1):
    #     colors.append(str("#") + '%06X' % randint(0, 0xFFFFFF))
    # visual_style["vertex_color"] = [colors[vs_outdegree[vertex.index]] for vertex in g.vs()]

    colors = []
    for i in range(0, int(math.ceil ( max(vs_closeness) * 100 )) , 1):
        colors.append(str("#") + '%06X' % randint(0, 0xFFFFFF))
    visual_style["vertex_color"] = [colors[int(vs_closeness[vertex.index]*100)] for vertex in g.vs()]

    # colors = []
    # for i in range(0, int(math.ceil(max(vs_betweenness))+10), 10):
    #     colors.append(str("#") + '%06X' % randint(0, 0xFFFFFF))
    # visual_style["vertex_color"] = [colors[int(vs_betweenness[vertex.index] / 10)] for vertex in g.vs()]

    # Order vertices in bins based on outdegree
    # bins = np.linspace(0, max(vs_outdegree), len(colors))
    # digitized_degrees = np.digitize(vs_outdegree, bins)
    #
    # # Set colors according to bins
    # visual_style["vertex_color"] = [colors[x - 1] for x in digitized_degrees]

    # Set label size
    visual_style["vertex_label_size"] = 30

    # Set label position
    visual_style["vertex_label_dist"] = 1.2

    plot(g, "closeness.png", **visual_style)

def read_nodeshp(file):
    noderows = list()
    shape_data = gpd.read_file(file)
    for idx, row in shape_data.iterrows():
        # id = idx
        geometry = row["geometry"]
        node_type = row[1]
        level = row["level"]
        function = row["function"]
        number = row["number"]
        newrow = [idx, geometry, node_type, level, function, number]
        noderows.append(newrow)
    return noderows


def read_linkshp(file):
    linkrows = list()
    shape_data = gpd.read_file(file)
    for idx, row in shape_data.iterrows():
        # id = idx
        geometry = row["geometry"]
        T1 = row["T1"]
        T2 = row["T2"]
        level = row["level"]
        id_infloor = idx
        newrow = [idx, geometry, T1, T2, level]
        linkrows.append(newrow)
    return linkrows

def combine2floors(directory, floor1, combinednodes, combinedlinks):
    floor1_nodes_name = directory + "/" + "Floor_" + floor1 + "_node.shp"
    floor1_links_name = directory + "/" + "Floor_" + floor1 + "_link.shp"

    floor1_nodes = read_nodeshp(floor1_nodes_name)
    floor1_links = read_linkshp(floor1_links_name)

    if len(combinednodes) == 0:
        combinednodes = floor1_nodes
        combinedlinks = floor1_links
        return combinednodes, combinedlinks

    num = len(combinednodes)

    for floor1_node in floor1_nodes:
        floor1_node[0] += combinednodes[-1][0]
    combinednodes = combinednodes + floor1_nodes

    for floor1_link in floor1_links:
        floor1_link[2] = num + floor1_link[2]
        floor1_link[3] = num + floor1_link[3]
    combinedlinks = combinedlinks + floor1_links

    return combinednodes, combinedlinks


directory = "data/Y25_Network"

floor1 = "H"
floor2 = "J"
floor3 = "K"
floor4 = "L"

combinednodes = list()
combinedlinks = list()

combinednodes, combinedlinks = combine2floors(directory, floor1, combinednodes, combinedlinks)
combinednodes, combinedlinks = combine2floors(directory, floor2, combinednodes, combinedlinks)
combinednodes, combinedlinks = combine2floors(directory, floor3, combinednodes, combinedlinks)
combinednodes, combinedlinks = combine2floors(directory, floor4, combinednodes, combinedlinks)

building_graph = Graph()
# np_combinednodes = np.array(combinednodes)
# np_combinedlinks = np.array(combinedlinks)

building_graph.add_vertices(len(combinednodes))
for index in range(0, len(building_graph.vs)):
    building_graph.vs[index]["pid"] = combinednodes[index][0]
    building_graph.vs[index]["geometry"] = combinednodes[index][1]
    building_graph.vs[index]["type"] = combinednodes[index][2]
    building_graph.vs[index]["level"] = combinednodes[index][3]
    building_graph.vs[index]["function"] = combinednodes[index][4]
    building_graph.vs[index]["number"] = combinednodes[index][5]


linked_vslist = list()
# print(len(combinedlinks))
for link in combinedlinks:
    linked_vslist.append((int(link[2]), int(link[3])))

building_graph.add_edges(linked_vslist)
for index in range(0, len(building_graph.es)):
    building_graph.es[index]["geometry"] = combinedlinks[index][1]
    building_graph.es[index]["level"] = combinedlinks[index][4]
    # print(combinedlinks[index][1].length)
    building_graph.es[index]["length"] = combinedlinks[index][1].length

num = len(building_graph.es)

building_graph.add_edge(38, 187)
building_graph.es[len(building_graph.es)-1]["geometry"] = None
building_graph.es[len(building_graph.es)-1]["level"] = 1.5
building_graph.es[len(building_graph.es)-1]["length"] = 4

building_graph.add_edge(36, 186)
building_graph.es[len(building_graph.es)-1]["geometry"] = None
building_graph.es[len(building_graph.es)-1]["level"] = 1.5
building_graph.es[len(building_graph.es)-1]["length"] = 4

building_graph.add_edge(54, 214)
building_graph.es[len(building_graph.es)-1]["geometry"] = None
building_graph.es[len(building_graph.es)-1]["level"] = 1.5
building_graph.es[len(building_graph.es)-1]["length"] = 4

building_graph.add_edge(187, 345)
building_graph.es[len(building_graph.es)-1]["geometry"] = None
building_graph.es[len(building_graph.es)-1]["level"] = 2.5
building_graph.es[len(building_graph.es)-1]["length"] = 4

building_graph.add_edge(186, 344)
building_graph.es[len(building_graph.es)-1]["geometry"] = None
building_graph.es[len(building_graph.es)-1]["level"] = 2.5
building_graph.es[len(building_graph.es)-1]["length"] = 4

building_graph.add_edge(214, 372)
building_graph.es[len(building_graph.es)-1]["geometry"] = None
building_graph.es[len(building_graph.es)-1]["level"] = 2.5
building_graph.es[len(building_graph.es)-1]["length"] = 4


building_graph.add_edge(345, 493)
building_graph.es[len(building_graph.es)-1]["geometry"] = None
building_graph.es[len(building_graph.es)-1]["level"] = 3.5
building_graph.es[len(building_graph.es)-1]["length"] = 4

building_graph.add_edge(344, 495)
building_graph.es[len(building_graph.es)-1]["geometry"] = None
building_graph.es[len(building_graph.es)-1]["level"] = 3.5
building_graph.es[len(building_graph.es)-1]["length"] = 4

building_graph.add_edge(372, 539)
building_graph.es[len(building_graph.es)-1]["geometry"] = None
building_graph.es[len(building_graph.es)-1]["level"] = 3.5
building_graph.es[len(building_graph.es)-1]["length"] = 4


# print(building_graph.betweenness(30))
# print(building_graph.closeness(0))
# print(building_graph.closeness(10))
# print(building_graph.closeness(30))
#
# print(building_graph.betweenness(123))
# print(building_graph.betweenness(125))
# print(building_graph.closeness(123))
# print(building_graph.closeness(125))


print(building_graph.betweenness(38))
print(building_graph.betweenness(36))
print(building_graph.betweenness(54))
print(building_graph.betweenness(187))
print(building_graph.betweenness(186))
print(building_graph.betweenness(214))
print(building_graph.betweenness(345))
print(building_graph.betweenness(344))
print(building_graph.betweenness(372))
print(building_graph.betweenness(493))
print(building_graph.betweenness(495))
print(building_graph.betweenness(539))

#
# print(building_graph.closeness(38))
# print(building_graph.closeness(36))
# print(building_graph.closeness(54))
# print(building_graph.closeness(187))
# print(building_graph.closeness(186))
# print(building_graph.closeness(214))
# print(building_graph.closeness(345))
# print(building_graph.closeness(344))
# print(building_graph.closeness(372))
# print(building_graph.closeness(493))
# print(building_graph.closeness(495))
# print(building_graph.closeness(539))

# print(building_graph.betweenness(38, weights="length"))
# print(building_graph.betweenness(36, weights="length"))
# print(building_graph.betweenness(54, weights="length"))
# print(building_graph.betweenness(187, weights="length"))
# print(building_graph.betweenness(186, weights="length"))
# print(building_graph.betweenness(214, weights="length"))
# print(building_graph.betweenness(345, weights="length"))
# print(building_graph.betweenness(344, weights="length"))
# print(building_graph.betweenness(372, weights="length"))
# print(building_graph.betweenness(493, weights="length"))
# print(building_graph.betweenness(495, weights="length"))
# print(building_graph.betweenness(539, weights="length"))

# print(building_graph.betweenness(109, weights="length"))
# print(building_graph.betweenness(426, weights="length"))
# #
# print(building_graph.closeness(109, weights="length", mode= OUT))
# print(building_graph.closeness(426, weights="length", mode= OUT))
#
# print(building_graph.closeness(38, weights="length"))
# print(building_graph.closeness(36, weights="length"))
# print(building_graph.closeness(54, weights="length"))
# print(building_graph.closeness(187, weights="length"))
# print(building_graph.closeness(186, weights="length"))
# print(building_graph.closeness(214, weights="length"))
# print(building_graph.closeness(345, weights="length"))
# print(building_graph.closeness(344, weights="length"))
# print(building_graph.closeness(372, weights="length"))
# print(building_graph.closeness(493, weights="length"))
# print(building_graph.closeness(495, weights="length"))
# print(building_graph.closeness(539, weights="length"))

# path = building_graph.get_shortest_paths(1, 596, weights="length", mode = OUT )
# print(path)
# for index in path:
#     print(building_graph.vs[index]["type"])
#     print(building_graph.vs[index]["function"])
#     print(building_graph.vs[index]["level"])
#     print(building_graph.vs[index]["number"])

# cl = building_graph.community_fastgreedy()
# membership = cl.as_clustering().membership
# _plotbycd(building_graph, membership)

# print(building_graph.vs[38]["pid"], building_graph.vs[38]["type"], building_graph.vs[38]["level"], building_graph.vs[38]["function"], building_graph.vs[38]["number"])
# print(building_graph.vs[54]["pid"], building_graph.vs[54]["type"], building_graph.vs[54]["level"], building_graph.vs[54]["function"], building_graph.vs[54]["number"])
# print(building_graph.vs[36]["pid"], building_graph.vs[36]["type"], building_graph.vs[36]["level"], building_graph.vs[36]["function"], building_graph.vs[36]["number"])
#
# print(building_graph.vs[187]["pid"], building_graph.vs[187]["type"], building_graph.vs[187]["level"], building_graph.vs[187]["function"], building_graph.vs[187]["number"])
# print(building_graph.vs[214]["pid"], building_graph.vs[214]["type"], building_graph.vs[214]["level"], building_graph.vs[214]["function"], building_graph.vs[214]["number"])
# print(building_graph.vs[186]["pid"], building_graph.vs[186]["type"], building_graph.vs[186]["level"], building_graph.vs[186]["function"], building_graph.vs[186]["number"])
#
# print(building_graph.vs[345]["pid"], building_graph.vs[345]["type"], building_graph.vs[345]["level"], building_graph.vs[345]["function"], building_graph.vs[345]["number"])
# print(building_graph.vs[372]["pid"], building_graph.vs[372]["type"], building_graph.vs[372]["level"], building_graph.vs[372]["function"], building_graph.vs[372]["number"])
# print(building_graph.vs[344]["pid"], building_graph.vs[344]["type"], building_graph.vs[344]["level"], building_graph.vs[344]["function"], building_graph.vs[344]["number"])
#
# print(building_graph.vs[493]["pid"], building_graph.vs[493]["type"], building_graph.vs[493]["level"], building_graph.vs[493]["function"], building_graph.vs[493]["number"])
# print(building_graph.vs[539]["pid"], building_graph.vs[539]["type"], building_graph.vs[539]["level"], building_graph.vs[539]["function"], building_graph.vs[539]["number"])
# print(building_graph.vs[495]["pid"], building_graph.vs[495]["type"], building_graph.vs[495]["level"], building_graph.vs[495]["function"], building_graph.vs[495]["number"])

