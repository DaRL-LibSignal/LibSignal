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
        if node_id not in self.net_node_dict.keys():
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
    print("Cannot find the road id {0}".format(road_id))
    sys.exit(-1)
    # return None

def build_int_intersection_map(roadnet_file, save_dir="graph_info.pkl",node_degree=4):
    '''
    generate the map between int ---> intersection ,intersection --->int
    generate the map between int ---> roads,  roads ----> int
    generate the required adjacent matrix
    generate the degree vector of node (we only care the valid roads(not connect with virtual intersection), and intersections)
    return: map_dict, and adjacent matrix
    save res into save dir, the res is as below
    res = [net_node_dict_id2inter,net_node_dict_inter2id,net_edge_dict_id2edge,net_edge_dict_edge2id,
        node_degree_node,node_degree_edge,node_adjacent_node_matrix,node_adjacent_edge_matrix,
        edge_adjacent_node_matrix]
    '''
    roadnet_dict = json.load(open(roadnet_file,"r"))
    valid_intersection_id = [i["id"] for i in roadnet_dict["intersections"] if not i["virtual"]]
    net_node_dict_id2inter = {}
    net_node_dict_inter2id = {}
    net_edge_dict_id2edge = {}
    net_edge_dict_edge2id = {}
    node_degree_node = [] # the num of adjacent nodes of node
    node_degree_edge = [] # the valid num of adjacent edges of node
    node_adjacent_node_matrix = [] #adjacent node of each node
    node_adjacent_edge_matrix = [] #adjacent edge of each node
    edge_adjacent_node_matrix = [] #adjacent node of each edge
    invalid_roads = []
    cur_num = 0
    #build the map between id and intersection
    for node_dict in roadnet_dict["intersections"]:
        if node_dict["virtual"]:
            for i in node_dict["roads"]:
                invalid_roads.append(i)
            continue
        node_id = node_dict["id"]
        net_node_dict_id2inter[cur_num] = node_id
        net_node_dict_inter2id[node_id] = cur_num
        cur_num +=1
    #map between id and intersection built done
    if cur_num!=len(valid_intersection_id):
        print("cur_num={}".format(cur_num))
        print("valid_intersection_id length={}".format(len(valid_intersection_id)))
        raise ValueError("cur_num should equal to len(valid_intersection_id)")
    
    #build the map between id and intersection and built the adjacent matrix of edge (i.e. the
    # adjacent nodes of the edge)  directed edge:[input_node_id, output_node_id]
    cur_num=0
    for edge_dict in roadnet_dict["roads"]:
        edge_id = edge_dict["id"]
        if edge_id in invalid_roads:
            continue
        else:
            net_edge_dict_id2edge[cur_num] = edge_id
            net_edge_dict_edge2id[edge_id] = cur_num
            cur_num+=1
            input_node = edge_dict['startIntersection']
            output_node = edge_dict['endIntersection']
            input_node_id = net_node_dict_inter2id[input_node]
            output_node_id = net_node_dict_inter2id[output_node]
            edge_adjacent_node_matrix.append([input_node_id,output_node_id])
    
    # build adjacent matrix for node (i.e the adjacent node of the node, and the 
    # adjacent edge of the node)
    for node_dict in roadnet_dict["intersections"]:
        if node_dict["virtual"]:
            continue        
        node_id = node_dict["id"]
        road_links = node_dict['roads']
        input_nodes = [] #should be node_degree
        input_edges = [] # needed, should be node_degree
        for road_link_id in road_links:
            road_link_dict = get_road_dict(roadnet_dict,road_link_id)
            if road_link_dict['endIntersection'] == node_id:
                if road_link_id in net_edge_dict_edge2id.keys():
                    input_edge_id = net_edge_dict_edge2id[road_link_id]
                    input_edges.append(input_edge_id)
                else:
                    continue
                start_node = road_link_dict['startIntersection']
                if start_node in net_node_dict_inter2id.keys():
                    start_node_id = net_node_dict_inter2id[start_node]
                    input_nodes.append(start_node_id)
        if len(input_nodes)!=len(input_edges):
            print(len(input_nodes))
            print(len(input_edges))
            print(node_id)
            raise ValueError("len(input_nodes) should be equal to len(input_edges)")
        node_degree_node.append(len(input_nodes))
        node_degree_edge.append(len(input_edges))
        while len(input_nodes)<node_degree:
            input_nodes.append(0)
        while len(input_edges)<node_degree:
            input_edges.append(0)
        node_adjacent_edge_matrix.append(input_edges)
        node_adjacent_node_matrix.append(input_nodes)
        # net_node = {
        #     'node_id': node_id,
        #     'input_nodes': list(set(input_nodes)),
        #     'input_edges': list(set(input_edges)),
        #     'output_nodes': list(set(output_nodes)),
        #     'output_edges': output_edges# should be a dict, with key as an input edge, value as output edges
        # }

    node_degree_node = np.array(node_degree_node) # the num of adjacent nodes of node
    node_degree_edge = np.array(node_degree_edge) # the valid num of adjacent edges of node
    node_adjacent_node_matrix = np.array(node_adjacent_node_matrix) 
    node_adjacent_edge_matrix = np.array(node_adjacent_edge_matrix)
    edge_adjacent_node_matrix = np.array(edge_adjacent_node_matrix)

    res = [net_node_dict_id2inter,net_node_dict_inter2id,net_edge_dict_id2edge,net_edge_dict_edge2id,
        node_degree_node,node_degree_edge,node_adjacent_node_matrix,node_adjacent_edge_matrix,
        edge_adjacent_node_matrix]
    save_file = open(save_dir,"wb")
    pickle.dump(res,save_file)
    save_file.close()
    return res

def analyse_vehicle_nums(file_path):
    replay_buffer = pickle.load(open(file_path,"rb"))
    observation = [i[0] for i in replay_buffer]
    observation = np.array(observation)
    observation = observation.reshape([-1])
    print("the mean of vehicle nums is ", observation.mean())
    print("the max of vehicle nums is ", observation.max())
    print("the min of vehicle nums is ", observation.min())
    print("the std of vehicle nums is ", observation.std())




if __name__ == "__main__":
    build_int_intersection_map( "/mnt/c/users/onlyc/desktop/work/RRL_TLC/roadnet_atlanta.json", save_dir="graph_info_1x5.pkl")
    # analyse_type=1
    # # 0 means analyze the vehicle information
    # # 1 means analyze the attention matrix of colight
    # # 2 means analyze topology of roadnet graph
    # if analyse_type==0:
    #     file_path = "data/replay_data/ana33fb_half.pkl"
    #     analyse_vehicle_nums(file_path)
    # elif analyse_type==1:
    #     agent_num = 4
    #     print_sum = True
    #     #yzy8_Colight_syn33_0.001_0.8_0.9995_64_1000_5000_20200407-155418_colight-100_att_ana.pkl # 5 head #syn331
    #     #yzy9_Colight_syn33_0.001_0.8_0.9995_64_1000_5000_20200407-155519_colight-100_att_ana.pkl # 5 head #syn332
    #     #yzy8_Colight_syn33_0.001_0.8_0.9995_64_1000_5000_20200407-181821_colight-100_att_ana.pkl # 1 head #syn331
    #     # yzy24_Colight_hz44_0.001_0.8_0.9995_64_1000_5000_20200408-125837_colight-100_att_ana.pkl # hz443
    #     file_path = "data/analysis/colight/yzy9_Colight_syn33_0.001_0.8_0.9995_64_1000_5000_20200407-155519_colight-100_att_ana.pkl"
    #     # yzynettest_syn33_0.001_0.8_0.9995_64_1000_5000_20200410-230841netlight-100_att_ana.pkl # syn332
    #     #file_path = "data/analysis/netlight/yzynettest_syn33_0.001_0.8_0.9995_64_1000_5000_20200410-230841netlight-100_att_ana.pkl"
    #     attention_mat = pickle.load(open(file_path,"rb"))
    #     tmp = np.array(attention_mat[50])
    #     print(tmp.shape)
    #     for i in[50,100,150,199,250]:
    #         tmp = attention_mat[i][agent_num]
    #         if print_sum:
    #             tmp=np.array(tmp)
    #             tmp=np.sum(tmp,axis=0)
    #         print(tmp)
    #         #print(attention_mat[i][agent_num])
    #         print("-"*20)
    #     # print(attention_mat[100][agent_num])
    #     # print("*"*20)
    #     # print(attention_mat[150][agent_num])
    #     # print("*"*20)
    #     # print(attention_mat[199][agent_num])
    #     # print("*"*20)
    #     # print(attention_mat[250][agent_num])
    # elif analyse_type==2:
    #     config_dir = "data/roadnet/roadnet_syn_4_4.json"
    #     res = build_int_intersection_map(config_dir,save_dir="graph_info_syn22.pkl")
    #     net_node_dict_id2inter = res[0]
    #     net_node_dict_inter2id =res[1]
    #     net_edge_dict_id2edge=res[2]
    #     net_edge_dict_edge2id=res[3]
    #     node_degree_node=res[4]
    #     node_degree_edge=res[5]
    #     node_adjacent_node_matrix=res[6]
    #     node_adjacent_edge_matrix=res[7]
    #     edge_adjacent_node_matrix=res[8]
    #     print(net_node_dict_id2inter)
    #     print("-"*20)
    #     print(net_node_dict_inter2id)
    #     print("-"*20)
    #     print(net_edge_dict_id2edge)
    #     print("-"*20)
    #     print(net_edge_dict_edge2id)
    #     print("-"*20)
    #     print(node_degree_node)
    #     print("-"*20)
    #     print(node_degree_edge)
    #     print("-"*20)
    #     print(node_adjacent_node_matrix)
    #     print("-"*20)
    #     print(node_adjacent_edge_matrix)
    #     print("-"*20)
    #     print(edge_adjacent_node_matrix)
    #     print("-"*20)
    #     print("done")