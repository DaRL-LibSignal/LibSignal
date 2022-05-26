import json
import os.path as osp
import citypb
from common.registry import Registry

import json


class Intersection(object):
    def __init__(self,  )



@Registry.register_world('openengine')
class World(object):
    """
    Create a Citypb engine and maintain infromations about Citypb world
    """
    def __init__(self, citypb_config, thread_num):
        print("building world from cb engine...")
        self.eng = citypb.Engine(citypb_config, thread_num)
        


        self.RIGHT = Ture  # how to set right or left

    def _get_roadnet(citypb_config):
        """
        generate roadnet dictionary based on providec configuration file
        functions borrowed form openengine CBEngine.py
        Details:
        collect roadnet informations.
        {1-'intersections'-(len=N_intersections):
            {11-'id': name of the intersection,
             12-'point': 121: {'x', 'y'}(intersection at this position),
             13-'width': itersection width,
             14-'roads'(len=N_roads controled by this intersection): name of road
             15-'roadLinks'(len=N_road links): 
                {151-'type': diriction type(go_straight, turn_left, turn_right, turn_U),  # TODO: check turn_u
                 152-'startRoad': start road name,
                 153-'endRoad': end road name,
                 154-'direction': int(same as type)
                 155-'laneLinks(len-N_lane links of road): 
                    {1651-'startLaneIndex': int(lane index in start road),
                     1652-'endLaneIndex': int(lane index in end road),
                     1653-'points(N_points alone this lane': {'x', 'y'}(point pos)
                     }
                 }
             16-'trafficLight:
             17-'virtual': bool
             }
         2-'roads'-(len=N_roads ): 
            {21-'id': name of the road,
             22-'points': {221: {'x', 'y'}(start pos), 222: {'x', 'y'}(end pos)}',
             23-'lanes'-(N_lanes in this road):
                {231: {'width': lane width, 'maxSpeed': max speed of each car on this lane}
                 232: 'startIntersection': lane start,
                 233: 'endIntersection': lane end
                 }
             }
         }
        """
        result = dict({'intersections': [], 'roads': []})

        with open(citypb_config, 'r') as f:
            lines = f.readlines()
            cnt = 0
            pre_road = 0
            is_obverse = 0
            for line in lines:
                # preprocessing line formation
                line = line.rstrip('\n').split(' ')
                if ('' in line):
                    line.remove('')
                if (len(line) == 1):  # this is notation 
                    if cnt == 0:
                        agent_num = int(line[0])  # start intersection segment
                        cnt += 1
                    elif cnt == 1:
                        road_num = int(line[0]) * 2  # start road segment
                        cnt += 1
                    elif cnt == 2:
                        signal_num = int(line[0])  # start signal segment
                        cnt += 1
                else:
                    if cnt == 1:  # intersection processing
                        # id, point, width, roads, roadLinks, trafficLight, virtual
                        result['intersections'].append({
                            'id': int(line[2]), 'point': {'x': float(line[0]), 'y': float(line[1])},
                            # No width
                            
                            }
                        )
                        
