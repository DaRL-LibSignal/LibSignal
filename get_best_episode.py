import os

def extract_log(lst):
    with open(lst[0], 'r') as f:
        contents = f.readlines()
    test = list()
    limit = 200
    for l in contents:
        if limit == 0:
            break
        l = l.split(', ')
        if 'Test step' in l[0]:
            limit -= 1
            travel_time = float(l[1].split(':')[1])
            queue = float(l[4].split(':')[1])
            delay = float(l[5].split(':')[1])
            throughput = float(l[6].split(':')[1])
            test.append([l[0].split(':')[1],travel_time, queue, delay, throughput])
    res = sorted(test,key=lambda x: x[1])
    print("best episode:",res[0])


lst = [os.path.join(os.getcwd(),x) for x in ['data/output_data/tsc/ppo_pfrl/0/logger/ppo_pfrl_20220614-184322.log'] ]
extract_log(lst)
