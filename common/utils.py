import pickle
import numpy as np 
import json
import os
import sys
import yaml
import copy
import logging
from datetime import datetime

from common.registry import Registry
import world


def get_road_dict(roadnet_dict, road_id):
    for item in roadnet_dict['roads']:
        if item['id'] == road_id:
            return item
    raise KeyError("environment and graph setting mapping error, no such road exists")


def build_index_intersection_map(roadnet_file):
    """
    generate the map between identity ---> index ,index --->identity
    generate the map between int ---> roads,  roads ----> int
    generate the required adjacent matrix
    generate the degree vector of node (we only care the valid roads(not connect with virtual intersection), and intersections)
    return: map_dict, and adjacent matrix
    res = [net_node_dict_id2inter,net_node_dict_inter2id,net_edge_dict_id2edge,net_edge_dict_edge2id,
        node_degree_node,node_degree_edge,node_adjacent_node_matrix,node_adjacent_edge_matrix,
        edge_adjacent_node_matrix]
    """
    roadnet_dict = json.load(open(roadnet_file, "r"))
    virt = "virtual" # judge whether is virtual node, especially in convert_sumo file
    if "gt_virtual" in roadnet_dict["intersections"][0]:
        virt = "gt_virtual"
    valid_intersection_id = [node["id"] for node in roadnet_dict["intersections"] if not node[virt]]
    node_id2idx = {}
    node_idx2id = {}
    edge_id2idx = {}
    edge_idx2id = {}
    node_degrees = []  # the num of adjacent nodes of node

    edge_list = []  # adjacent node of each node
    node_list = []  # adjacent edge of each node
    sparse_adj = []  # adjacent node of each edge
    invalid_roads = []
    cur_num = 0
    # build the map between identity and index of node
    for node_dict in roadnet_dict["intersections"]:
        if node_dict[virt]:
            for node in node_dict["roads"]:
                invalid_roads.append(node)
            continue
        cur_id = node_dict["id"]
        node_idx2id[cur_num] = cur_id
        node_id2idx[cur_id] = cur_num
        cur_num += 1
    # map between identity and index built done

    # sanity check of node number equals intersection numbers
    if cur_num != len(valid_intersection_id):
        raise ValueError("environment and graph setting mapping error, node 1 to 1 mapping error")
    
    # build the map between identity and index and built the adjacent matrix of edge
    cur_num = 0
    for edge_dict in roadnet_dict["roads"]:
        edge_id = edge_dict["id"]
        if edge_id in invalid_roads:
            continue
        else:
            edge_idx2id[cur_num] = edge_id
            edge_id2idx[edge_id] = cur_num
            cur_num += 1
            input_node_id = edge_dict['startIntersection']
            output_node_id = edge_dict['endIntersection']
            input_node_idx = node_id2idx[input_node_id]
            output_node_idx = node_id2idx[output_node_id]
            sparse_adj.append([input_node_idx, output_node_idx])
    
    # build adjacent matrix for node (i.e the adjacent node of the node, and the 
    # adjacent edge of the node)
    for node_dict in roadnet_dict["intersections"]:
        if node_dict[virt]:
            continue        
        node_id = node_dict["id"]
        road_links = node_dict['roads']
        input_nodes = []  # should be node_degree
        input_edges = []  # needed, should be node_degree
        for road_link_id in road_links:
            road_link_dict = get_road_dict(roadnet_dict, road_link_id)
            if road_link_dict['endIntersection'] == node_id:
                if road_link_id in edge_id2idx.keys():
                    input_edge_idx = edge_id2idx[road_link_id]
                    input_edges.append(input_edge_idx)
                else:
                    continue
                start_node = road_link_dict['startIntersection']
                if start_node in node_id2idx.keys():
                    start_node_idx = node_id2idx[start_node]
                    input_nodes.append(start_node_idx)
        if len(input_nodes) != len(input_edges):
            raise ValueError(f"{node_id} : number of input node and edge not equals")
        node_degrees.append(len(input_nodes))

        edge_list.append(input_edges)
        node_list.append(input_nodes)

    node_degrees = np.array(node_degrees)  # the num of adjacent nodes of node
    sparse_adj = np.array(sparse_adj)  # the valid num of adjacent edges of node

    result = {'node_idx2id': node_idx2id, 'node_id2idx': node_id2idx,
              'edge_idx2id': edge_idx2id, 'edge_id2idx': edge_id2idx,
              'node_degrees': node_degrees, 'sparse_adj': sparse_adj,
              'node_list': node_list, 'edge_list': edge_list}
    return result



def analyse_vehicle_nums(file_path):
    replay_buffer = pickle.load(open(file_path, "rb"))
    observation = [i[0] for i in replay_buffer]
    observation = np.array(observation)
    observation = observation.reshape([-1])
    print("the mean of vehicle nums is ", observation.mean())
    print("the max of vehicle nums is ", observation.max())
    print("the min of vehicle nums is ", observation.min())
    print("the std of vehicle nums is ", observation.std())

