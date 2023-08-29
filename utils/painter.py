import matplotlib.pyplot as plt
import numpy as np


def info_helper(line, position):
    processed = line.strip('\n').split('\t')
    result = [float(processed[i]) for i in position]
    result.append(processed[1])
    return result

def info_updater(record, arrival):
    record[k].append(v)

def painter(files, subscribers):
    # subscriber order is 0: epoch, 1: average travel time, 
    # 2: q loss, 3: rewards, 4: queue, 5: delay, 6: throughput
    fig = plt.figure(figsize=(3 * len(subscribers)-1, 8), constrained_layout=True)
    rows = fig.subfigures(2, 1)
    cols = []
    for idx, row in enumerate(rows):
        col = row.subplots(1, len(subscribers)-1)
        cols.append(col)
    mapping = {'epoch': 2, 'average travel time': 3, 'q_loss': 4, 'rewards':5, 'queue':6, 'delay':7, 'throughput':8}
    try:
        subscribers_id = [mapping[item] for item in subscribers]
    except KeyError as e:
        raise NotImplementedError(f' {e} subscriber is not implemented')

    for j, file in enumerate(files.keys()):
        train_records = []
        test_records = []
        with open(files[file], 'r') as f:
            contents = f.readlines()
        for line in contents:
            info = info_helper(line, subscribers_id)
            train_records.append(info[:-1]) if info[-1] =='TRAIN' else \
                test_records.append(info[:-1])
        train_data = np.array(train_records)
        test_data = np.array(test_records)
        label = ['TRAIN', 'TEST']
        val = [train_data, test_data]
        color = ['#448ee4', '#1fa774']
        for idx, row in enumerate(rows):
            row.suptitle(label[idx])
            data = val[idx]
            for i, ax in enumerate(cols[idx],1):
                ax.plot(i-1, 1)
                ax.plot(data[:, 0], data[:, i], label=f'{file}',color=color[j])
                ax.set_xlabel(f'{subscribers[0]}')
                ax.set_ylabel(f'{subscribers[i]}')
                ax.legend()
        
            # for i, ax in enumerate(cols,1):
            #     ax.plot(i-1, 1)
            #     ax.plot(data[:, 0], data[:, i], label=f'{file}')
            #     ax.set_xlabel(f'{subscribers[0]}')
            #     ax.set_ylabel(f'{subscribers[i]}')
            #     ax.legend()
    plt.show()
    # plt.savefig('test.png', format='png')

if __name__ == '__main__':
    files = {'sumo': '/home/jovyan/DaRL/LibSignal/data/output_data/tsc/sumo_dqn/sumohz1x1/test/logger/2023_08_28-23_39_55_DTL.log', 'cityflow':'/home/jovyan/DaRL/LibSignal/data/output_data/tsc/cityflow_dqn/cityflow1x1/test/logger/2023_08_28-21_34_43_DTL.log'}
    painter(files, ['epoch', 'average travel time', 'rewards','delay'])