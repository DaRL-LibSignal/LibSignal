import os, sys
from wrapper import Wrapper
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import citypb

def main():
    roadnet_file = os.path.join(os.getcwd(), 'data/raw_data/pb_hangzhou/hangzhou.txt')
    flow_file =  os.path.join(os.getcwd(), 'data/raw_data/pb_hangzhou/0_flow.txt')
    wrapper = Wrapper(roadnet_file, flow_file)
    wrapper.test()
    pass


if __name__ == '__main__':
    main()
