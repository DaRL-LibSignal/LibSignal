import os, sys
from wrapper import Wrapper
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import citypb

def main():
    roadnet_file = '/data/raw_data/pb_novel/roadnet_novel.txt'
    flow_file = '/data/raw_data/pb_nanchang/flow_novel.txt'
    wrapper = Wrapper(roadnet_file, flow_file)
    wrapper.test()
    pass

if __name__ == '__main__':
    main()