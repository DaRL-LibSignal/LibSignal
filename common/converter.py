"""
convert SUMO file to CityFlow file.
"""

from ast import operator
import os
from random import random
import sys
from sys import platform
import argparse
from collections import defaultdict
import sympy
from mpmath import degrees, radians
import copy
import math
import json
import xml.etree.cElementTree as ET
import xml.dom.minidom
from itertools import groupby
from operator import itemgetter
from functools import cmp_to_key

if platform == "linux" or platform == "linux2":
    # this is linux
    try:
        import traci
        import traci.constants as tc
        import sumolib
        from sumolib.net import Connection
    except ImportError:
        if "SUMO_HOME" in os.environ:
            print(os.path.join(os.environ["SUMO_HOME"], "tools"))
            sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
            import traci
            import traci.constants as tc
            import sumolib
            from sumolib.net import Connection
        else:
            raise EnvironmentError(
                "Please set SUMO_HOME environment variable or install traci as python module!")
elif platform == "win32":
    os.environ['SUMO_HOME'] = 'D:\\software\\sumo-0.32.0'
    try:
        import traci
        import traci.constants as tc
        import sumolib
        from sumolib.net import Connection
    except ImportError:
        if "SUMO_HOME" in os.environ:
            print(os.path.join(os.environ["SUMO_HOME"], "tools"))
            sys.path.append(
                os.path.join(os.environ["SUMO_HOME"], "tools")
            )
            import traci
            import traci.constants as tc
            import sumolib
            from sumolib.net import Connection
        else:
            raise EnvironmentError(
                "Please set SUMO_HOME environment variable or install traci as python module!")

elif platform == 'darwin':
    os.environ['SUMO_HOME'] = "/Users/{0}/sumo/".format(os.environ.get('USER'))
    print(os.environ['SUMO_HOME'])
    try:
        import traci
        import traci.constants as tc
        import sumolib
        from sumolib.net import Connection
    except ImportError:
        if "SUMO_HOME" in os.environ:
            print(os.path.join(os.environ["SUMO_HOME"], "tools"))
            sys.path.append(
                os.path.join(os.environ["SUMO_HOME"], "tools")
            )
            import traci
            import traci.constants as tc
            import sumolib
            from sumolib.net import Connection
        else:
            raise EnvironmentError(
                "Please set SUMO_HOME environment variable or install traci as python module!")
else:
    sys.exit("platform error")


def parse_args():
    parser = argparse.ArgumentParser()
    # sumo2cityflow
    # parser.add_argument("--or_sumonet", type=str,
    #                     default='grid4x4/grid4x4.net.xml')
    # parser.add_argument("--cityflownet", type=str,
    #                     default='grid4x4/mygrid4x4_roadnet_red.json')
    # parser.add_argument("--or_sumoflow", type=str,
    #                     default='grid4x4/grid4x4.rou.xml')
    # parser.add_argument("--cityflowflow", type=str,
    #                     default='grid4x4/mygrid4x4_flow.json')
    # parser.add_argument("--sumocfg", type=str,
    #                     default='grid4x4/grid4x4.sumocfg')

    # parser.add_argument("--or_sumonet", type=str,
    #                     default='cologne1/cologne1.net.xml')
    # parser.add_argument("--cityflownet", type=str,
    #                     default='cologne1/cologne1_roadnet_red.json')
    # parser.add_argument("--or_sumoflow", type=str,
    #                     default='cologne1/cologne1.rou.xml')
    # parser.add_argument("--cityflowflow", type=str,
    #                     default='cologne1/cologne1_flow.json')
    # parser.add_argument("--sumocfg", type=str,
    #                     default='cologne1/cologne1.sumocfg')

    # parser.add_argument("--or_sumonet", type=str,
    #                     default='cologne3/cologne3.net.xml')
    # parser.add_argument("--cityflownet", type=str,
    #                     default='cologne3/cologne3_roadnet_red.json')
    # parser.add_argument("--or_sumoflow", type=str,
    #                     default='cologne3/cologne3.rou.xml')
    # parser.add_argument("--cityflowflow", type=str,
    #                     default='cologne3/cologne3_flow.json')
    # parser.add_argument("--sumocfg", type=str,
    #                     default='cologne3/cologne3.sumocfg')

    # parser.add_argument("--or_sumonet", type=str,
    #                     default='ingolstadt21/ingolstadt21.net.xml')
    # parser.add_argument("--cityflownet", type=str,
    #                     default='ingolstadt21/ingolstadt21_roadnet_red.json')
    # parser.add_argument("--or_sumoflow", type=str,
    #                     default='ingolstadt21/ingolstadt21.rou.xml')
    # parser.add_argument("--cityflowflow", type=str,
    #                     default='ingolstadt21/ingolstadt21_flow.json')
    # parser.add_argument("--sumocfg", type=str,
    #                     default='ingolstadt21/ingolstadt21.sumocfg')

    # cityflow2sumo
    # parser.add_argument("--or_cityflownet", type=str,
    #                     default='hangzhou_1x1_bc-tyc_18041610_1h/roadnet.json')
    # parser.add_argument("--sumonet", type=str,
    #                     default='hangzhou_1x1_bc-tyc_18041610_1h/hangzhou_1x1_bc-tyc_18041610_1h_test_test.net.xml')
    # parser.add_argument("--or_cityflowflow", type=str,
    #                     default='hangzhou_1x1_bc-tyc_18041610_1h/flow.json')
    # parser.add_argument("--sumoflow", type=str,
    #                     default='hangzhou_1x1_bc-tyc_18041610_1h/hangzhou_1x1_bc-tyc_18041610_1h.rou.xml')

    # parser.add_argument("--or_cityflownet", type=str,
    #                     default='hangzhou_4x4_gudang_18041610_1h/roadnet_4X4.json')
    # parser.add_argument("--sumonet", type=str,
    #                     default='hangzhou_4x4_gudang_18041610_1h/hangzhou_4x4_gudang_18041610_1h.net.xml')
    # parser.add_argument("--or_cityflowflow", type=str,
    #                     default='hangzhou_4x4_gudang_18041610_1h/hangzhou_4x4_gudang_18041610_1h.json')
    # parser.add_argument("--sumoflow", type=str,
    #                     default='hangzhou_4x4_gudang_18041610_1h/hangzhou_4x4_gudang_18041610_1h.rou.xml')

    parser.add_argument("--or_cityflownet", type=str,
                        default='atlanta_1x5/roadnet_atlanta.json')
    parser.add_argument("--sumonet", type=str,
                        default='atlanta_1x5/atlanta_1x5.net.xml')
    parser.add_argument("--or_cityflowflow", type=str,
                        default='atlanta_1x5/atlanta.json')
    parser.add_argument("--sumoflow", type=str,
                        default='atlanta_1x5/atlanta_1x5.rou.xml')

    return parser.parse_args()


U_TURN_AS = "turn_left"
DEBUG = False
TRUE_CORRECTION_lane = True
SUMO_PROGRAM = True


def get_direction_fron_connection(connection):
    _map = {
        Connection.LINKDIR_STRAIGHT: "go_straight",
        Connection.LINKDIR_TURN: "turn_u",
        Connection.LINKDIR_LEFT: "turn_left",
        Connection.LINKDIR_RIGHT: "turn_right",
        Connection.LINKDIR_PARTLEFT: "turn_left",
        Connection.LINKDIR_PARTRIGHT: "turn_right",
    }
    return _map[connection.getDirection()]


def process_edge(edge):
    lanes = []
    if TRUE_CORRECTION_lane:
        for inx, lane in enumerate(reversed(edge.getLanes())):
            outgoing_list = lane.getOutgoing()
            for outgoing in outgoing_list:
                new_lane = copy.copy(lane)
                direction = get_direction_fron_connection(outgoing)
                to_lane = outgoing.getToLane()
                # marky,add to_lane
                new_lane._cityflow_lane_id = f'{lane.getID()}|{to_lane.getID()}|{direction}'
                new_lane._cityflow_lane_inx = inx
                new_lane._direction = direction
                lanes.append(new_lane)
            if len(outgoing_list) == 0:
                new_lane = copy.copy(lane)
                new_lane._cityflow_lane_id = f'{lane.getID()}'
                new_lane._cityflow_lane_inx = inx
                new_lane._direction = 'go_end'
                lanes.append(new_lane)
    else:
        for lane in edge.getLanes():
            outgoing_list = lane.getOutgoing()
            for outgoing in outgoing_list:
                new_lane = copy.copy(lane)
                direction = get_direction_fron_connection(outgoing)
                to_lane = outgoing.getToLane()
                new_lane._cityflow_lane_id = f'{lane.getID()}|{to_lane.getID()}|{direction}'
                new_lane._direction = direction
                lanes.append(new_lane)
            if len(outgoing_list) == 0:
                new_lane = copy.copy(lane)
                new_lane._cityflow_lane_id = f'{lane.getID()}'
                new_lane._direction = 'go_end'
                lanes.append(new_lane)
    edge._cityflow_lanes = lanes[::-1]
    return edge


def _cityflow_get_lane_index_in_edge(lane, edge):
    for i, _lane in enumerate(edge._cityflow_lanes):
        if _lane._cityflow_lane_id == lane._cityflow_lane_id:
            return i
    raise Exception('lane in edge not found')


def _cityflow_get_lane_index_in_edge_cor(lane, edge):
    ## i = lane._cityflow_lane_id.split('|')[0]
    for i, _lane in enumerate(edge._cityflow_lanes):
        if _lane._cityflow_lane_id == lane._cityflow_lane_id:
            return _lane._cityflow_lane_inx
    raise Exception('lane in edge not found')


def point_tuple_to_dict(point_tuple):
    return {"x": point_tuple[0], "y": point_tuple[1]}


def _is_node_virtual(node,tls_dict):
    n = node
    edges = [edge for edge in n.getIncoming() + n.getOutgoing()]
    ids = list(set([e.getFromNode().getID()
               for e in edges] + [e.getToNode().getID() for e in edges]))
    # for virtual node belongs to dead_end,it just has 2 roads, non-virtual has at least 3 roads.
    # if len(ids) <= 2 or node.getID() not in tls_dict:
    if len(ids) <= 2 or (node.getID() not in tls_dict and 'GS_'+node.getID() not in tls_dict):
    # if (node.getID() not in tls_dict) and ('GS_'+node.getID() not in tls_dict):
        return True
    else:
        return False


def group_connections_by_start_end(connections):
    connection_group_result = defaultdict(list)
    for connection in connections:
        start_road = connection.getFrom()
        end_road = connection.getTo()
        direction = get_direction_fron_connection(connection)
        key = "{}|{}|{}".format(
            start_road.getID(), end_road.getID(), direction)
        connection_group_result[key].append(connection)
    return connection_group_result


def calc_edge_compass_angle(edge):
    north_ray = sympy.Ray((0, 0), (0, 1))
    # 反向算进入edge，直观。
    edge_ray = sympy.Ray(*edge.getShape()[:2][::-1])
    angle = north_ray.closing_angle(edge_ray)
    angle = (angle + 2 * sympy.pi) % (2 * sympy.pi)
    angle_degrees = float(degrees(angle))
    angle_radians = float(radians(degrees(angle)))
    edge._angle_degrees = round(angle_degrees, 4)
    edge._angle_radians = round(angle_radians, 4)
    return angle_degrees, angle_radians


def calc_edge_compass_angle_no_modify(edge):
    north_ray = sympy.Ray((0, 0), (0, 1))
    # 要算所有edge，所以不要反向。
    edge_ray = sympy.Ray(*edge.getShape()[:2])
    angle = north_ray.closing_angle(edge_ray)
    angle = (angle + 2 * sympy.pi) % (2 * sympy.pi)
    angle_degrees = float(degrees(angle))
    # angle_radians = float(radians(degrees(angle)))
    # edge._angle_degrees = round(angle_degrees,4)
    # edge._angle_radians = round(angle_radians,4)
    return angle_degrees


def process_intersection_simple_phase(intersection):
    if intersection['virtual']:
        return intersection

    all_green = {
        "time": 30,
        "availableRoadLinks": intersection['trafficLight']['roadLinkIndices']
    }
    all_red = {
        "time": 30,
        "availableRoadLinks": []
    }
    lightphases = [all_green]
    intersection['trafficLight']['lightphases'] = lightphases
    return intersection


def _cal_angle_pair(cluster):
    centroids = cluster['centroids']
    centroids = [x[0] for x in centroids]
    if len(centroids) == 4:
        pairs = [(centroids[0], centroids[2]), (centroids[1], centroids[3])]
    elif len(centroids) == 3:
        r1 = centroids[1] - centroids[0]
        r2 = centroids[2] - centroids[0]
        r3 = centroids[2] - centroids[1]
        near180_1 = abs(180 - r1)
        near180_2 = abs(180 - r2)
        near180_3 = abs(180 - r3)
        lista = [
            ([(centroids[0], centroids[1]), (centroids[2],)], near180_1),
            ([(centroids[0], centroids[2]), (centroids[1],)], near180_2),
            ([(centroids[0],), (centroids[1], centroids[2]), ], near180_3),
        ]
        pairs = min(lista, key=lambda item: item[1])[0]
    elif len(centroids) == 2:
        pairs = [(centroids[0], centroids[1]), ]
    elif len(centroids) == 1:
        pairs = [(centroids[0],), ]
    return pairs


def find_edges_by_angle(all_edges, angle):
    edges = []
    for edge in all_edges:
        if math.isclose(edge._angle_degrees, angle, abs_tol=0.0001):
            # if edge._angle_degrees == angle:
            edges.append(edge)
    if not edges:
        raise Exception('!!!no edge._angle_degrees = angle')
    return edges


def find_edges_by_cluster_centroid(all_edges, angle):
    edges = []
    for edge in all_edges:
        if math.isclose(edge._cluster_centroid[0], angle, abs_tol=0.0001):
            # if edge._angle_degrees == angle:
            edges.append(edge)
    if not edges:
        raise Exception('!!!no edge._cluster_centroid[0] = angle')
    return edges


def get_all_turn_right_link_index(roadLinks):
    allow = []
    for index, roadlink in enumerate(roadLinks):
        if roadlink['type'] == 'turn_right':
            allow.append(index)
    return allow


def filter_roadlinks_by_startedge_and_turn_type(roadLinks, edge, turntype):
    result = []
    for index, roadlink in enumerate(roadLinks):
        if roadlink['startRoad'] == edge.getID() and roadlink['type'] == turntype:
            result.append((index, roadlink))
    return result


def filter_roadlinks_by_startedge(roadLinks, lane_id):
    result = []
    edge_id, lane_index = lane_id.split('_')
    for index, roadlink in enumerate(roadLinks):
        lane_index_list = []
        for laneLink in roadlink['laneLinks']:
            lane_index_list.append(laneLink['startLaneIndex'])
        lane_index_list = list(set(lane_index_list))

        if roadlink['startRoad'] == edge_id and int(lane_index) in lane_index_list:
            result.append((index, roadlink))
    return result


def fill_empty_phase(current_phase, count):
    need_fill_count = count - len(current_phase)
    for x in range(need_fill_count):
        empty_phase_dict = {
            'availableRoadLinks': [],
            'time': 0,
        }
        current_phase.append(empty_phase_dict)
    return current_phase


all_phase_dict = {}
node_outgoing_dict = {}


def node_to_intersection(node, tls_dict, edge_dict):
    node_type = node.getType()
    node_coord = node.getCoord()
    intersection = {
        "id": node.getID(),
        "point": {"x": node_coord[0], "y": node_coord[1]},
        # warning.road width is undefined for any intersection.default 15
        "width": 0 if _is_node_virtual(node, tls_dict) else 15,
        # "width": 0 if (_is_node_virtual(node, tls_dict) and node_type not in ['priority']) else 15,
        "roads": [edge.getID() for edge in node.getIncoming() + node.getOutgoing()],

        # "_roads":[{'id':}]
        "roadLinks": [],
        "trafficLight": {
            "roadLinkIndices": [],
            "lightphases": []
        },
        "virtual": False,
        "gt_virtual": _is_node_virtual(node,tls_dict),  # dead_end判断为virtual
    }

    connections_group = group_connections_by_start_end(node.getConnections())
    roadLinks = intersection['roadLinks']
    for k, v in connections_group.items():
        connection_template = v[0]
        start_road = connection_template.getFrom()
        end_road = connection_template.getTo()
        # 加上驶入方向的正北夹角
        raw_roadlink_type = get_direction_fron_connection(connection_template)
        roadLink = {
            "type": raw_roadlink_type,
            "startRoad": start_road.getID(),
            "endRoad": end_road.getID(),
            "direction": 0,  # TODO: direction is falsely defined but it doesn't affect usage
            "laneLinks": []
        }
        if roadLink["type"] == "turn_u":
            roadLink["type"] = U_TURN_AS

        for start_lane in reversed(start_road._cityflow_lanes):
            if start_lane._direction != raw_roadlink_type:
                continue
            # TODO lane enumerate
            if TRUE_CORRECTION_lane:
                for end_inx, end_lane in enumerate(reversed(end_road._lanes)):
                    start_point = start_lane.getShape()[-1]
                    start_point = point_tuple_to_dict(start_point)
                    end_point = end_lane.getShape()[0]
                    end_point = point_tuple_to_dict(end_point)
                    path = {
                        "startLaneIndex": _cityflow_get_lane_index_in_edge_cor(start_lane, start_road),
                        "endLaneIndex": end_inx,
                        # ytodo: 或许改为起始lane结束点，路口点，结束lane起始点。
                        "points": [start_point, end_point]  # warning 飞行模式
                    }
                    roadLink["laneLinks"].append(path)
            else:
                for end_lane in end_road._cityflow_lanes:
                    start_point = start_lane.getShape()[-1]
                    start_point = point_tuple_to_dict(start_point)
                    end_point = end_lane.getShape()[0]
                    end_point = point_tuple_to_dict(end_point)
                    path = {
                        "startLaneIndex": _cityflow_get_lane_index_in_edge(start_lane, start_road),
                        "endLaneIndex": _cityflow_get_lane_index_in_edge(end_lane, end_road),
                        "points": [start_point, end_point]  # warning 飞行模式
                    }
                    roadLink["laneLinks"].append(path)
        roadLinks.append(roadLink)

    for i, _ in enumerate(intersection["roadLinks"]):
        intersection["trafficLight"]["roadLinkIndices"].append(i)

    if node_type in ['dead_end']:
        pass
    if node_type in ['priority']:
        pass
    if node_type in ['right_before_left']:
        pass
    if node_type in ['dead_end', 'priority', 'right_before_left']:
        intersection = process_intersection_simple_phase(intersection)

    if node_type in ['traffic_light', 'traffic_light_right_on_red']:
        print(node.getID())
        if SUMO_PROGRAM:
            all_phase = []
            nodeid = node.getID()
            tlnodeid = nodeid
            all_phase_dict[nodeid] = []
            G_to_lane_dict = {}
            if nodeid not in tls_dict.keys():
                tlnodeid = 'GS_' + tlnodeid
            for connec in tls_dict[tlnodeid]._connections:
                G_to_lane_dict[connec[-1]] = connec[0].getID()

            for idx_phase in tls_dict[tlnodeid]._programs['0']._phases:
                phase, duration = idx_phase.state, idx_phase.duration
                lane_list = []
                for i, alpha in enumerate(phase):
                    if (alpha == 'G' or alpha == 'g') and i in G_to_lane_dict.keys():
                        lane_list.append(G_to_lane_dict[i])

                lane_list_ = []
                for lane in lane_list:
                    edge_id, lane_id = lane.split('_')
                    lane_id = int(lane_id)
                    lane_ = edge_id + '_' + \
                        str(len(edge_dict[edge_id])-lane_id-1)
                    lane_list_.append(lane_)

                all_phase_dict[nodeid].append(list(set(lane_list_)))
                index_list = []

                for _lane in lane_list_:
                    index_roadlink_list = filter_roadlinks_by_startedge(
                        roadLinks, _lane)
                    index_list += [item[0] for item in index_roadlink_list]
                phase_dict = {
                    'availableRoadLinks': list(set(index_list)),
                    'time': duration
                }
                all_phase.append(phase_dict)
            intersection["trafficLight"]["lightphases"] = all_phase

            outgoing_lane_list = []
            edge_list_ = [edge_.getID() for edge_ in node.getOutgoing()]
            for edge in edge_list_:
                for i in range(len(edge_dict[edge])):
                    outgoing_lane_list.append(edge+'_'+str(i))
            node_outgoing_dict[nodeid] = outgoing_lane_list

        exiting_lane_list = []
        for edge in node.getOutgoing():
            exiting_lane_list.extend([lane.getID()
                                     for lane in edge.getLanes()])

    return intersection


def get_final_intersections(net, tls_dict, edge_dict):

    final_intersections = []
    net_nodes = net.getNodes()
    net_nodes_sorted = sorted(net_nodes, key=lambda n: n.getID())
    nodes = [(index, node) for index, node in enumerate(net_nodes_sorted)]
    nodes = nodes[:]
    for obj in nodes:

        index = obj[0]
        node = obj[1]

        intersection = node_to_intersection(node, tls_dict, edge_dict)
        if intersection["roads"] != []:
            final_intersections.append(intersection)

    return final_intersections

def get_final_roads(net):
    edges = net.getEdges()
    final_roads = []
    for edge in edges:
        start_intersection = edge.getFromNode()
        start_coord = start_intersection.getCoord()
        end_intersection = edge.getToNode()
        end_coord = end_intersection.getCoord()
        tmp_points = edge.getShape()
        points = []
        # points.append({"x":start_coord[0],"y":start_coord[1]})
        # length_p = len(tmp_points)
        # for i in range(length_p):
        #     points.append({"x":tmp_points[i][0],"y":tmp_points[i][1]})
        # points.append({"x":end_coord[0],"y":end_coord[1]})
        points.append({"x":start_coord[0],"y":start_coord[1]})
        length_p = len(tmp_points)
        for i in range(1,length_p-1):
            points.append({"x":tmp_points[i][0],"y":tmp_points[i][1]})
        points.append({"x":end_coord[0],"y":end_coord[1]})
        road = {
            "id": edge.getID(),
            "points": points,
            "lanes": [
            ],
            "startIntersection": start_intersection.getID(),
            "endIntersection": end_intersection.getID(),
        }
        if DEBUG:
            road['_compass_angle'] = calc_edge_compass_angle_no_modify(edge)
        if TRUE_CORRECTION_lane:
            for _v in edge._lanes:
                road["lanes"].append({
                    "width": _v._width,
                    "maxSpeed": _v._speed
                })
        else:
            for _v in edge._cityflow_lanes:
                road["lanes"].append({
                    "width": _v._width,
                    "maxSpeed": _v._speed
                })
        final_roads.append(road)
    return final_roads


def sumo2cityflow_flow(args):
    # parent dir of current dir
    f_cwd = os.path.abspath(os.path.dirname(os.getcwd()) + os.path.sep + ".")
    sumofile = os.path.join(f_cwd, 'data/raw_data', args.or_sumoflow)
    cityflowfile = os.path.join(f_cwd, 'data/raw_data', args.cityflowflow)

    sumocfg = os.path.join(f_cwd, 'data/raw_data', args.sumocfg)
    print("Converting sumo flow file", sumofile)
    tree = ET.parse(sumofile)
    root = tree.getroot()
    # TODO check whether need to get truth rou.xml from trip.xml
    if root.find('trip') != None and 'rou' in sumofile:
        # 1. rename sumoflowfile from rou.xml to trip.xml
        src = sumofile
        dst = get_filename(sumofile, typ='trip')
        try:
            os.rename(src, dst)
        except Exception as e:
            print(e)
            print('rename file fail\r\n')
        else:
            print('rename file success\r\n')
        # 2. create true rou.xml
        sumofile = get_filename(sumofile, typ='rou')
        sumonet = os.path.join(f_cwd, 'data/raw_data', args.or_sumonet)
        os.system(
            f"duarouter --route-files={dst} --net-file={sumonet} --output-file={sumofile}")
        print("SUMO rou file generated successfully!")
    
    # redirect tree of sumoflow file
    tree = ET.parse(sumofile)
    root = tree.getroot()
    tree_cfg = ET.parse(sumocfg)
    root_cfg = tree_cfg.getroot()
    start_time = int(root_cfg.find('time').find('begin').attrib['value'])
    end_time = int(root_cfg.find('time').find('end').attrib['value'])
    assert end_time-start_time == 3600
    flows = []
    length = 5.0
    width = 1.8
    maxPosAcc = 2.6
    maxNegAcc = 4.5
    minGap = 2.5
    if root.find('vType') != None:
        length = float(root.find('vType').attrib['length']) if 'length' in root.find('vType').attrib else 5.0
        width = float(root.find('vType').attrib['width']) if 'width' in root.find('vType').attrib else 1.8
        maxPosAcc = float(root.find('vType').attrib['accel']) if 'accel' in root.find('vType').attrib else 2.6
        maxNegAcc = float(root.find('vType').attrib['decel']) if 'decel' in root.find('vType').attrib else 4.5
        minGap = float(root.find('vType').attrib['minGap']) if 'minGap' in root.find('vType').attrib else 2.5
    for obj in root.iter('vehicle'):
        routes = (obj.find('route').attrib['edges']).split()
        if len(routes) < 2:
            continue
        flows.append({
            "vehicle": {
                "length": length,
                "width": width,
                "maxPosAcc": maxPosAcc,
                "maxNegAcc": maxNegAcc,
                "usualPosAcc": maxPosAcc,
                "usualNegAcc": maxNegAcc,
                "minGap": minGap,
                "maxSpeed": 13.39,
                "headwayTime": 1.5
            },
            "route": routes,
            "interval": 5.0,
            "startTime": int(float(obj.attrib['depart'])) - start_time,
            "endTime": int(float(obj.attrib['depart'])) - start_time
        })
    with open(cityflowfile, "w") as f:
        json.dump(flows, f)
    print("Cityflow flow file generated successfully!")


def sumo2cityflow_net(args):
    # parent dir of current dir
    f_cwd = os.path.abspath(os.path.dirname(os.getcwd()) + os.path.sep + ".")
    sumofile = os.path.join(f_cwd, 'data/raw_data', args.or_sumonet)
    cityflowfile = os.path.join(f_cwd, 'data/raw_data', args.cityflownet)
    print("Converting sumo net file", args.or_sumonet)
    net = sumolib.net.readNet(sumofile, withPrograms=True)

    for edge in net.getEdges():
        process_edge(edge)

    tls_dict = {}
    for tls in net.getTrafficLights():
        tls_dict[tls.getID()] = tls

    print('Start processing '+str(len(tls_dict))+" traffic lights")
    edge_dict = {}
    for edge_ in net.getEdges():
        edge_dict[edge_.getID()] = edge_._lanes

    final_intersections = get_final_intersections(net, tls_dict, edge_dict)

    final_roads = get_final_roads(net)

    result = {
        "intersections": final_intersections,
        "roads": final_roads
    }

    f = open(cityflowfile, 'w')
    json.dump(result, f, indent=2)
    f.close()
    print("Cityflow net file generated successfully!")


def cityflow2sumo_flow(args):
    # parent dir of current dir
    f_cwd = os.path.abspath(os.path.dirname(os.getcwd()) + os.path.sep + ".")
    sumofile = os.path.join(f_cwd, 'data/raw_data', args.sumoflow)
    cityflowfile = os.path.join(f_cwd, 'data/raw_data', args.or_cityflowflow)

    data = json.load(open(cityflowfile, 'r', encoding="utf-8"))
    # sorted vehicle according to depart time
    data = sorted(data,key=lambda x:x['startTime'])

    doc = xml.dom.minidom.Document()
    root = doc.createElement('routes')
    root.setAttribute('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
    root.setAttribute('xsi:noNamespaceSchemaLocation',
                      'http://sumo.dlr.de/xsd/routes_file.xsd')
    doc.appendChild(root)
    # set general info
    node_vtype = doc.createElement('vType')
    node_vtype.setAttribute('id', 'pkw')
    node_vtype.setAttribute('length', '5.0')
    node_vtype.setAttribute('width', '2.0')
    node_vtype.setAttribute('minGap', '2.5')
    node_vtype.setAttribute('maxSpeed', '11.111')
    node_vtype.setAttribute('accel', '2.0')
    node_vtype.setAttribute('decel', '4.5')
    root.appendChild(node_vtype)

    for idx, info in enumerate(data):
        vehicle, route, interval, startTime, endTime = info['vehicle'], info[
            'route'], info['interval'], info['startTime'], info['endTime']
        node_vehicle = doc.createElement('vehicle')
        node_vehicle.setAttribute('id', str(idx))
        node_vehicle.setAttribute('depart', str(startTime))
        # node_vehicle.setAttribute('departLane', 'best')
        # node_vehicle.setAttribute('departSpeed', 'max')

        node_route = doc.createElement('route')
        node_route.setAttribute('edges', ' '.join(route))

        node_vehicle.appendChild(node_route)
        root.appendChild(node_vehicle)

    fp = open(sumofile, 'w')
    doc.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding="utf-8")
    print("SUMO flow file generated successfully!")

def get_start_idx(lists):
    new_lists = {}
    for key, value in lists.items():
        k,v = list(value.keys())[0],list(value.values())[0]
        start_idx = sum([x_v for _,x in lists.items() for x_k,x_v in x.items() if x_k<k])
        new_lists[key] = (start_idx,v)
    return new_lists

def cmp_turn_direction(x,y):
    '''
    Used for sort 4 types of turn direction operations: turn-r,turn-s,turn-l,(optional)turn-u
    False: less,
    True: greater
    '''
    if x['type'] =='turn_left' and y['type'] =='turn_left': # one of them is turn-u
        # judge which is turn-l, which is turn-u
        if (x['startRoad'] == '-'+x['endRoad']) or (x['endRoad'] == '-'+x['startRoad']): # x is turn-u
            return 1
        elif (y['startRoad'] == '-'+y['endRoad']) or (y['endRoad'] == '-'+y['startRoad']):
            return -1
    elif x['type'] == 'turn_right':
        return -1
    elif y['type'] == 'turn_right':
        return 1
    elif x['type'] == 'turn_straight':
        return -1
    elif y['type'] == 'turn_straight':
        return 1
    elif x['type'] =='turn_left':
        return 1
    else:
        return -1

def judg_turn_u(x, data):
    '''
    In Cityflow, turn_u is allowed but present as turn_left.
    So, judging whether turn_left belongs to turn_u is necessary for generating tlLogic in SUMO.
    '''
    start_info = []
    end_info = []
    count = 0
    for i in data:
        if i['id'] == x['startRoad']:
            start_info.append(i['startIntersection'])
            start_info.append(i['endIntersection'])
            count += 1
        if i['id'] == x['endRoad']:
            end_info.append(i['startIntersection'])
            end_info.append(i['endIntersection'])
            count += 1
        if count == 2:
            break
    if start_info[1] == end_info[0] and start_info[0] == end_info[1]:
        return True # turn_u
    return False

def cityflow2sumo_net(args):
    """generate net.xml according to nod.xml, edg.xml,con.xml,tll.xml"""
    # parent dir of current dir
    f_cwd = os.path.abspath(os.path.dirname(os.getcwd()) + os.path.sep + ".")
    sumofile = os.path.join(f_cwd, 'data/raw_data', args.sumonet)
    cityflowfile = os.path.join(f_cwd, 'data/raw_data', args.or_cityflownet)

    sumo_node = get_filename(sumofile, 'nod')
    sumo_edge = get_filename(sumofile, 'edg')
    sumo_con = get_filename(sumofile, 'con')
    sumo_tll = get_filename(sumofile, 'tll')

    data = json.load(open(cityflowfile, 'r', encoding="utf-8"))

    # generate nod.xml, con.xml and tll.xml
    doc_node = xml.dom.minidom.Document()
    root_node = doc_node.createElement('nodes')
    root_node.setAttribute(
        'xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
    root_node.setAttribute('xsi:noNamespaceSchemaLocation',
                           'http://sumo.dlr.de/xsd/nodes_file.xsd')
    doc_node.appendChild(root_node)

    doc_con = xml.dom.minidom.Document()
    root_con = doc_con.createElement('connections')
    root_con.setAttribute('version', '1.1')
    root_con.setAttribute(
        'xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
    root_con.setAttribute('xsi:noNamespaceSchemaLocation',
                          'http://sumo.dlr.de/xsd/connections_file.xsd')
    doc_con.appendChild(root_con)

    doc_tll = xml.dom.minidom.Document()
    root_tll = doc_tll.createElement('tlLogics')
    root_tll.setAttribute('version', '1.1')
    root_tll.setAttribute(
        'xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
    root_tll.setAttribute('xsi:noNamespaceSchemaLocation',
                          'http://sumo.dlr.de/xsd/tllogic_file.xsd')
    doc_tll.appendChild(root_tll)

    for inter in data['intersections']:
        # nod.xml
        node = doc_node.createElement('node')
        node.setAttribute('id', inter['id'])
        node.setAttribute('x', str(inter['point']['x']))
        node.setAttribute('y', str(inter['point']['y']))
        node.setAttribute('type', str('priority')
                          if inter['virtual'] else 'traffic_light_right_on_red')
        root_node.appendChild(node)

        # group roads according to startRoad.
        road_group = []
        sortorder = {"turn_right":0, "go_straight":1, "turn_left":2, "turn_u":3}
        for idx, items in groupby(inter['roadLinks'], key=itemgetter('startRoad')):
            # sort roads according to order: turn right, turn straight, turn left, turn u
            l_items = list(items)
            for x in l_items:
                if x['type'] == 'turn_left':
                    if judg_turn_u(x, data['roads']):
                        x.update({'type':'turn_u'})
            sorted_items = sorted(l_items,key=lambda x: sortorder[x['type']])
            road_group.insert(0,sorted_items) # items' order is the same order as settings of SUMO: NESW
        # sort roads according to order: turn right, turn straight, turn left, turn u
        road_group = [y for x in road_group for y in x]
        phase_dic = {}
        for idx, x in enumerate(inter['roadLinks']):
            dst_idx = road_group.index(x)
            phase_dic[idx] = dict({dst_idx:len(road_group[dst_idx]['laneLinks'])})
        phase_dic = get_start_idx(phase_dic)
        phase_num_all = sum(len(i['laneLinks']) for i in inter['roadLinks'])
        # num_phase = len(inter['roadLinks'])
        # dic_phase = [0] * num_phase
        for idx, link in enumerate(inter['roadLinks']):
            # con.xml
            # num_lanelinks = len(link['laneLinks'])-1
            start_num_lanelinks = [len(x['lanes']) for x in data['roads'] if x['id']==link['startRoad']][0]-1
            end_num_lanelinks = [len(x['lanes']) for x in data['roads'] if x['id']==link['endRoad']][0]-1
            for lanelink in link['laneLinks']:
                con = doc_con.createElement('connection')
                con.setAttribute('from', link['startRoad'])
                con.setAttribute('to', link['endRoad'])
                # sumo (outer to inner) is opposite from cityflow(inner to outer) in lane order.
                con.setAttribute('fromLane', str(
                    abs(start_num_lanelinks-lanelink['startLaneIndex'])))
                con.setAttribute('toLane', str(
                    abs(end_num_lanelinks-lanelink['endLaneIndex'])))
                root_con.appendChild(con)
        if not inter['virtual']:
            # tll.xml,in cityflow setting, only conside G and r phase
            tll = doc_node.createElement('tlLogic')
            tll.setAttribute('id', inter['id'])
            tll.setAttribute('type', 'static')
            tll.setAttribute('programID', '0')
            tll.setAttribute('offset', '0')
            yellow_state = ['r'] * phase_num_all
            # yellow_state = ['r'] * num_phase * lane_num
            # yellow_state = ['r'] * num_phase * 3
            for idx,light in enumerate(inter['trafficLight']['lightphases']):
                state = ['r'] * phase_num_all
                # state = ['r'] * num_phase * lane_num
                # state = ['r'] * num_phase * 3
                if idx != 0 and light['availableRoadLinks'] is not None: # idx=0 means yellow phase
                    for act_roadlink in light['availableRoadLinks']:
                        state[phase_dic[act_roadlink][0]:sum(phase_dic[act_roadlink])] = ['G']*phase_dic[act_roadlink][1]
                        
                        # act_p = dic_phase[inter['trafficLight']['roadLinkIndices'][act_roadlink]]
                        # state[act_p*lane_num:act_p*lane_num+lane_num] = ['G']*lane_num
                        
                        # if num_phase == 12:
                        #     state[act_p*3:act_p*3+3] = ['G']*3
                        # elif num_phase == 8:
                        #     state[act_p*2:act_p*2+2] = ['G']*2
                    # add yellow phase behind green phase
                    phase = doc_node.createElement('phase')
                    phase.setAttribute('duration', str(light['time']))
                    phase.setAttribute('state', ''.join(state))
                    tll.appendChild(phase)
                    root_tll.appendChild(tll)
                    # add yellow phase behind green phase
                    phase_y = doc_node.createElement('phase')
                    phase_y.setAttribute('duration', '5')
                    phase_y.setAttribute('state', ''.join(yellow_state))
                    tll.appendChild(phase_y)
                    root_tll.appendChild(tll)
                if light['time'] <= 5:
                    # first should set yellow phase, then can add this yellow phase when adding a green phase
                    assert idx == 0
                    for act_roadlink in light['availableRoadLinks']:
                        yellow_state[phase_dic[act_roadlink][0]:sum(phase_dic[act_roadlink])] = ['G']*phase_dic[act_roadlink][1]
                        # act_p = dic_phase[inter['trafficLight']['roadLinkIndices'][act_roadlink]]
                        # state[act_p*lane_num:act_p*lane_num+lane_num] = ['G']*lane_num
                        
                        # if num_phase == 12:
                        #     yellow_state[act_p*3:act_p*3+3] = ['G']*3
                        # elif num_phase == 8:
                        #     yellow_state[act_p*2:act_p*2+2] = ['G']*2


        # if not inter['virtual']:
        #     # lane_type = len(set([i['type'] for i in inter['roadLinks']]))
        #     # group roads according to startRoad.
        #     road_group = []
        #     sortorder = {"turn_right":0, "go_straight":1, "turn_left":2, "turn_u":3}
        #     for idx, items in groupby(inter['roadLinks'], key=itemgetter('startRoad')):
        #         # sort roads according to order: turn right, turn straight, turn left, turn u
        #         l_items = list(items)
        #         for x in l_items:
        #             if x['type'] == 'turn_left':
        #                 if judg_turn_u(x, data['roads']):
        #                     x.update({'type':'turn_u'})
        #         sorted_items = sorted(l_items,key=lambda x: sortorder[x['type']])
        #         road_group.insert(0,sorted_items) # items' order is the same order as settings of SUMO: NESW
        #     # sort roads according to order: turn right, turn straight, turn left, turn u
        #     road_group = [y for x in road_group for y in x]
        #     phase_dic = {}
        #     for idx, x in enumerate(inter['roadLinks']):
        #         dst_idx = road_group.index(x)
        #         phase_dic[idx] = dict({dst_idx:len(road_group[dst_idx]['laneLinks'])})
        #     phase_dic = get_start_idx(phase_dic)
        #     phase_num_all = sum(len(i['laneLinks']) for i in inter['roadLinks'])
        #     # num_phase = len(inter['roadLinks'])
        #     # dic_phase = [0] * num_phase
        #     for idx, link in enumerate(inter['roadLinks']):
        #         # con.xml
        #         # num_lanelinks = len(link['laneLinks'])-1
        #         start_num_lanelinks = [len(x['lanes']) for x in data['roads'] if x['id']==link['startRoad']][0]-1
        #         end_num_lanelinks = [len(x['lanes']) for x in data['roads'] if x['id']==link['endRoad']][0]-1
        #         for lanelink in link['laneLinks']:
        #             con = doc_con.createElement('connection')
        #             con.setAttribute('from', link['startRoad'])
        #             con.setAttribute('to', link['endRoad'])
        #             # sumo (outer to inner) is opposite from cityflow(inner to outer) in lane order.
        #             con.setAttribute('fromLane', str(
        #                 abs(start_num_lanelinks-lanelink['startLaneIndex'])))
        #             con.setAttribute('toLane', str(
        #                 abs(end_num_lanelinks-lanelink['endLaneIndex'])))
        #             root_con.appendChild(con)
                
        #         # # create dic for {direction,type} and idx(roadLinks)
        #         # phase_idx = get_phase2lane(link['direction'], link['type'], num_phase)
        #         # dic_phase[inter['trafficLight']['roadLinkIndices'][idx]] = phase_idx
            
        #     # tll.xml,in cityflow setting, only conside G and r phase
        #     tll = doc_node.createElement('tlLogic')
        #     tll.setAttribute('id', inter['id'])
        #     tll.setAttribute('type', 'static')
        #     tll.setAttribute('programID', '0')
        #     tll.setAttribute('offset', '0')
        #     yellow_state = ['r'] * phase_num_all
        #     # yellow_state = ['r'] * num_phase * lane_num
        #     # yellow_state = ['r'] * num_phase * 3
        #     for idx,light in enumerate(inter['trafficLight']['lightphases']):
        #         state = ['r'] * phase_num_all
        #         # state = ['r'] * num_phase * lane_num
        #         # state = ['r'] * num_phase * 3
        #         if idx != 0 and light['availableRoadLinks'] is not None: # idx=0 means yellow phase
        #             for act_roadlink in light['availableRoadLinks']:
        #                 state[phase_dic[act_roadlink][0]:sum(phase_dic[act_roadlink])] = ['G']*phase_dic[act_roadlink][1]
                        
        #                 # act_p = dic_phase[inter['trafficLight']['roadLinkIndices'][act_roadlink]]
        #                 # state[act_p*lane_num:act_p*lane_num+lane_num] = ['G']*lane_num
                        
        #                 # if num_phase == 12:
        #                 #     state[act_p*3:act_p*3+3] = ['G']*3
        #                 # elif num_phase == 8:
        #                 #     state[act_p*2:act_p*2+2] = ['G']*2
        #             # add yellow phase behind green phase
        #             phase = doc_node.createElement('phase')
        #             phase.setAttribute('duration', str(light['time']))
        #             phase.setAttribute('state', ''.join(state))
        #             tll.appendChild(phase)
        #             root_tll.appendChild(tll)
        #             # add yellow phase behind green phase
        #             phase_y = doc_node.createElement('phase')
        #             phase_y.setAttribute('duration', '5')
        #             phase_y.setAttribute('state', ''.join(yellow_state))
        #             tll.appendChild(phase_y)
        #             root_tll.appendChild(tll)
        #         if light['time'] <= 5:
        #             # first should set yellow phase, then can add this yellow phase when adding a green phase
        #             assert idx == 0
        #             for act_roadlink in light['availableRoadLinks']:
        #                 yellow_state[phase_dic[act_roadlink][0]:sum(phase_dic[act_roadlink])] = ['G']*phase_dic[act_roadlink][1]
        #                 # act_p = dic_phase[inter['trafficLight']['roadLinkIndices'][act_roadlink]]
        #                 # state[act_p*lane_num:act_p*lane_num+lane_num] = ['G']*lane_num
                        
        #                 # if num_phase == 12:
        #                 #     yellow_state[act_p*3:act_p*3+3] = ['G']*3
        #                 # elif num_phase == 8:
        #                 #     yellow_state[act_p*2:act_p*2+2] = ['G']*2

                
    fp_node = open(sumo_node, 'w')
    doc_node.writexml(fp_node, addindent='\t', newl='\n', encoding="utf-8")
    fp_node.close()

    fp_con = open(sumo_con, 'w')
    doc_con.writexml(fp_con, addindent='\t', newl='\n', encoding="utf-8")
    fp_con.close()

    fp_tll = open(sumo_tll, 'w')
    doc_tll.writexml(fp_tll, addindent='\t', newl='\n', encoding="utf-8")
    fp_tll.close()

    print("SUMO node, connections and tll files generated successfully!")

    # generate edg.xml
    doc_edge = xml.dom.minidom.Document()
    root_edge = doc_edge.createElement('edges')
    root_edge.setAttribute(
        'xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
    root_edge.setAttribute('xsi:noNamespaceSchemaLocation',
                           'http://sumo.dlr.de/xsd/edges_file.xsd')
    doc_edge.appendChild(root_edge)
    for road in data['roads']:
        edge = doc_edge.createElement('edge')
        edge.setAttribute('id', road['id'])
        edge.setAttribute('from', road['startIntersection'])
        edge.setAttribute('to', road['endIntersection'])
        edge.setAttribute('numLanes', str(len(road['lanes'])))
        edge.setAttribute('speed', '11.11')
        edge.setAttribute('priority', '-1')
        root_edge.appendChild(edge)
    fp_edge = open(sumo_edge, 'w')
    doc_edge.writexml(fp_edge, indent='\t', addindent='\t',
                      newl='\n', encoding="utf-8")
    fp_edge.close()
    print("SUMO edge file generated successfully!")
    os.system(
        f"netconvert --node-files={sumo_node} --edge-files={sumo_edge} \
            --connection-files={sumo_con} --tllogic-files={sumo_tll} --output-file={sumofile}")
    print("SUMO net file generated successfully!")
# duarouter -n /home/lxl/TSCtest/LibSignalSpare_new/data/raw_data/cologne1/cologne1.net.xml -f /home/lxl/TSCtest/LibSignalSpare_new/data/raw_data/cologne1/cologne1_flows.xml -o /home/lxl/TSCtest/LibSignalSpare_new/data/raw_data/cologne1/cologne1_1_rou.xml

def get_phase2lane(direction, typ, num_phase):
    """
    return the idx of phase_idx related to roadlink_idx.
    In SUMO, the phase are ordered by clockwise, from N_r_s_l to W_r_s_l, totally 12 dims or 8dims.
    TODO maybe inregular eg:16
    """
    # N
    if direction == 3:
        if typ == 'turn_right':
            return 0
        elif typ == 'go_straight':
            return 1 if num_phase == 12 else 0
        else:
            return 2 if num_phase == 12 else 1
    # E
    elif direction == 2:
        if typ == 'turn_right':
            return 3
        elif typ == 'go_straight':
            return 4 if num_phase == 12 else 2
        else:
            return 5 if num_phase == 12 else 3
    # S
    elif direction == 1:
        if typ == 'turn_right':
            return 6
        elif typ == 'go_straight':
            return 7 if num_phase == 12 else 4
        else:
            return 8 if num_phase == 12 else 5
    # W
    else:
        if typ == 'turn_right':
            return 9
        elif typ == 'go_straight':
            return 10 if num_phase == 12 else 6
        else:
            return 11 if num_phase == 12 else 7

def get_filename(netfile, typ='', need_path=True):
    """
    typ: net,nod,edg,tll,rou,sumocfg
    """
    filepath, filename_all = os.path.split(netfile)
    filename = filename_all.split('.')
    if typ !='sumocfg':
        if need_path:
            file_res = os.path.join(filepath, filename[0]+'.'+typ+'.xml')
        else:
            file_res = filename[0]+'.'+typ+'.xml'
    else:
        file_res = os.path.join(filepath, filename[0]+'.sumocfg')
    return file_res

def cityflow2sumo_cfg(args):
    """generate sumo cfg file"""
    # parent dir of current dir
    f_cwd = os.path.abspath(os.path.dirname(os.getcwd()) + os.path.sep + ".")
    sumofile = os.path.join(f_cwd, 'data/raw_data', args.sumonet)
    sumo_cfg = get_filename(sumofile, typ='sumocfg')
    sumo_net = get_filename(sumofile, typ='net', need_path=False)
    sumo_route = get_filename(sumofile, typ='rou', need_path=False)

    doc = xml.dom.minidom.Document()
    root = doc.createElement('configuration')
    doc.appendChild(root)

    # input file:net.xml, route.xml
    input_file = doc.createElement('input')
    input_net = doc.createElement('net-file')
    input_net.setAttribute('value', sumo_net)
    input_file.appendChild(input_net)
    input_route = doc.createElement('route-files')
    input_route.setAttribute('value', sumo_route)
    input_file.appendChild(input_route)
    root.appendChild(input_file)
    # time:begin, end, default: 3600
    time = doc.createElement('time')
    begin = doc.createElement('begin')
    begin.setAttribute('value', '0')
    time.appendChild(begin)
    end = doc.createElement('end')
    end.setAttribute('value', '3600')
    time.appendChild(end)
    root.appendChild(time)

    fp = open(sumo_cfg, 'w')
    doc.writexml(fp, addindent='\t', newl='\n', encoding="utf-8")
    fp.close()

    print("SUMO cfg file generated successfully!")

if __name__ == '__main__':
    args = parse_args()
    # sumo2cityflow
    # sumo2cityflow_net(args)
    # sumo2cityflow_flow(args)

    # cityflow2sumo
    cityflow2sumo_net(args)
    # cityflow2sumo_flow(args)
    # cityflow2sumo_cfg(args)
