import os, sys
from wrapper1x1 import Wrapper
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import citypb

def main():
    roadnet_file = os.path.join(os.getcwd(), 'data/raw_data/pb1x1/roadnet_1x1.txt')
    flow_file =  os.path.join(os.getcwd(), 'data/raw_data/pb1x1/flow_1x1.txt')
    wrapper = Wrapper(roadnet_file, flow_file)
    wrapper.test()
    pass


if __name__ == '__main__':
    main()
