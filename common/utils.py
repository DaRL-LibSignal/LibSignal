import pickle
import numpy as np 
import json
import sys
import pandas as pd 
import os
import time
from copy import deepcopy


def generate_node_dict(roadnet_file):
    '''
    node dict has key as node id, value could be the dict of input nodes and output nodes
    :return:
    '''
    roadnet_dict = json.load(open(roadnet_file,"r"))
    net_node_dict = {}
    for node_dict in roadnet_dict["intersections"]:
        node_id = node_dict["id"]
        road_links = node_dict['roads']
        #input_roads_links = [i in node_dict["roads"] if i.beginwith("road_"+node_id[13:])]
        input_nodes = []
        output_nodes = [] #needed ,usually the same with input nodes
        input_edges = [] # needed 
        output_edges = {} # actually useless in Netlight
        for road_link_id in road_links:
            road_link_dict = get_road_dict(roadnet_dict["roads"],road_link_id)
            if road_link_dict['startIntersection'] == node_id:
                end_node = road_link_dict['endIntersection']
                output_nodes.append(end_node)
                # todo add output edges
            elif road_link_dict['endIntersection'] == node_id:
                input_edges.append(road_link_id)
                start_node = road_link_dict['startIntersection']
                input_nodes.append(start_node)
                output_edges[road_link_id] = set() # paris, roadlinks [road_in,road_out]
                pass
        # update roadlinks
        actual_roadlinks = node_dict['roadLinks']
        for actual_roadlink in actual_roadlinks:
            output_edges[actual_roadlink['startRoad']].add(actual_roadlink['endRoad'])

        net_node = {
            'node_id': node_id,
            'input_nodes': list(set(input_nodes)),
            'input_edges': list(set(input_edges)),
            'output_nodes': list(set(output_nodes)),
            'output_edges': output_edges# should be a dict, with key as an input edge, value as output edges
        }
        if node_id not in net_node_dict.keys():
            net_node_dict[node_id] = net_node
    #actually we have to give an int id to them in order to use in tf
    return net_node_dict


def generate_edge_dict(roadnet_file, net_node_dict):
    '''
    edge dict has key as edge id, value could be the dict of input edges and output edges
    :return:
    '''
    roadnet_dict = json.load(open(roadnet_file,"r"))
    net_edge_dict = {}
    for edge_dict in roadnet_dict['roads']:
        edge_id = edge_dict['id']
        input_node = edge_dict['startIntersection']
        output_node = edge_dict['endIntersection']

        net_edge = {
            'edge_id': edge_id,
            'input_node': input_node,
            'output_node': output_node,
            'input_edges': net_node_dict[input_node]['input_edges'],
            'output_edges': net_node_dict[output_node]['output_edges'][edge_id],
        }
        if edge_id not in net_edge_dict.keys():
            net_edge_dict[edge_id] = net_edge
    return net_edge_dict


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
    valid_intersection_id = [node["id"] for node in roadnet_dict["intersections"] if not node["virtual"]]
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
        if node_dict["virtual"]:
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
        if node_dict["virtual"]:
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


def load_config_dict(config_path):
    result = json.load(open(config_path, 'r'))
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


def get_output_file_path(task, model, prefix):
    path = os.path.join('./data/output_data', task, model, prefix)
    return path
