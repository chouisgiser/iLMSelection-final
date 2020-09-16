from src.processing.features.structural.building_graph import BuildingGraph
from src.processing.features.structural.building_graph import Floor
from utils import file_io

directory = '../../../data/features/structural/Y25_Network'

floor_id_list = ["H", "J", "K", "L"]
transition_pairs = [{"floor_id": (1,2),"vertex_ids": (38,187)}, {"floor_ids": (1,2),"vertex_ids": (36, 186)},
                  {"floor_ids": (1,2),"vertex_ids": (54, 214)}, {"floor_ids": (2,3),"vertex_ids": (187, 345)},
                  {"floor_ids": (2,3),"vertex_ids": (186, 344)}, {"floor_ids": (2,3),"vertex_ids": (214, 372)},
                  {"floor_ids": (3,4),"vertex_ids": (345, 493)}, {"floor_ids": (3,4),"vertex_ids": (344, 495)},
                  {"floor_ids": (3,4),"vertex_ids": (372, 539)}]


buildingGraph = BuildingGraph('Y25')
for floor_id in  floor_id_list:
    floor = Floor(floor_id)
    floor.read_node_shp(directory)
    floor.read_link_shp(directory)
    buildingGraph.add_floor(floor)

for transition_pair in transition_pairs :
    buildingGraph.add_links_between_floors(transition_pair)


elevator_ids = [38, 187, 345, 493]
candidate_list = ['H38', 'H67', 'H79', 'H92',
                  'J10', 'J39', 'J41', 'K13',
                  'K22', 'L11', 'L12', 'L26',
                  'L40', 'H5A', 'J5A', 'K5A',
                  'L5A']
# closeness
# filename = 'output/closeness.csv'
# dicts = list()
# headers = ["name", "closeness"]
#
# ele_closeness = buildingGraph.get_room_closeness(elevator_ids)
# row = {"name": 'Elevator', "closeness": ele_closeness}
# dicts.append(row)
#
# for candidate in candidate_list:
#     print(candidate)
#     vids = buildingGraph.get_id_by_number(candidate)
#     candidate_closeness = buildingGraph.get_room_closeness(vids)
#     row = {"name": candidate, "closeness": candidate_closeness}
#     dicts.append(row)
#
# file_io.write_csv_file(filename, dicts, headers)


# betweenness
# filename = 'output/betweenness.csv'
# # dicts = list()
# # headers = ["name", "betweenness"]
# #
# # ele_betweenness = buildingGraph.get_room_betweenness(elevator_ids)
# # row = {"name": 'Elevator', "betweenness": ele_betweenness}
# # dicts.append(row)
# #
# # for candidate in candidate_list:
# #     print(candidate)
# #     vids = buildingGraph.get_id_by_number(candidate)
# #     candidate_betweenness = buildingGraph.get_room_betweenness(vids)
# #     row = {"name": candidate, "betweenness": candidate_betweenness}
# #     dicts.append(row)
# #
# # file_io.write_csv_file(filename, dicts, headers)

# control
# filename = 'output/control_1.csv'
# dicts = list()
# headers = ["name", "control"]
#
# ele_control = buildingGraph.get_room_control(elevator_ids, "lift")
# row = {"name": 'Elevator', "control": ele_control}
# dicts.append(row)
#
# for candidate in candidate_list:
#     print(candidate)
#     vids = buildingGraph.get_id_by_number(candidate)
#     candidate_control = buildingGraph.get_room_control(vids, candidate)
#     row = {"name": candidate, "control": candidate_control}
#     dicts.append(row)
#
# file_io.write_csv_file(filename, dicts, headers)

# proximity2DP
filename = '../output/proximity2DP.csv'
dicts = list()
headers = ["name", "dis2DP"]
ele_dis2dp = buildingGraph.get_proximity_2_dp(elevator_ids)
row = {"name": 'Elevator', "dis2DP": ele_dis2dp}
dicts.append(row)

for candidate in candidate_list:
    print(candidate)
    vids = buildingGraph.get_id_by_number(candidate)
    dis2DP = buildingGraph.get_proximity_2_dp(vids)
    row = {"name": candidate, "dis2DP": dis2DP}
    dicts.append(row)

file_io.write_csv_file(filename, dicts, headers)
#
# # proximity2FE
filename = '../output/proximity2FE.csv'
dicts = list()
headers = ["name", "dis2FE"]
ele_dis2dp = buildingGraph.get_proximity_2_fe(elevator_ids)
row = {"name": 'Elevator', "dis2FE": ele_dis2dp}
dicts.append(row)

for candidate in candidate_list:
    print(candidate)
    vids = buildingGraph.get_id_by_number(candidate)
    dis2DP = buildingGraph.get_proximity_2_fe(vids)
    row = {"name": candidate, "dis2FE": dis2DP}
    dicts.append(row)

file_io.write_csv_file(filename, dicts, headers)
#
# # proximity2BE
filename = '../output/proximity2BE.csv'
dicts = list()
headers = ["name", "dis2BE"]
ele_dis2dp = buildingGraph.get_proximity_2_be(elevator_ids)
row = {"name": 'Elevator', "dis2BE": ele_dis2dp}
dicts.append(row)

for candidate in candidate_list:
    print(candidate)
    vids = buildingGraph.get_id_by_number(candidate)
    dis2DP = buildingGraph.get_proximity_2_be(vids)
    row = {"name": candidate, "dis2BE": dis2DP}
    dicts.append(row)

file_io.write_csv_file(filename, dicts, headers)

# vid = buildingGraph.get_id_by_number("J10")
# dis = buildingGraph.shortest_paths_dijkstra(165, 101, weights= "length")
# print(dis)
# path = buildingGraph.get_shortest_paths(165, 101, weights= "length")
# for index in path:
#     print(buildingGraph.vs[index]["vid_in_floor"])
#     print(buildingGraph.vs[index]["type"])
#     print(buildingGraph.vs[index]["function"])
#     print(buildingGraph.vs[index]["level"])
#     print(buildingGraph.vs[index]["number"])

# print(buildingGraph.betweenness(38))
# print(buildingGraph.betweenness(36))
# print(buildingGraph.betweenness(54))
# print(buildingGraph.betweenness(187))
# print(buildingGraph.betweenness(186))
# print(buildingGraph.betweenness(214))
# print(buildingGraph.betweenness(345))
# print(buildingGraph.betweenness(344))
# print(buildingGraph.betweenness(372))
# print(buildingGraph.betweenness(493))
# print(buildingGraph.betweenness(495))
# print(buildingGraph.betweenness(539))
