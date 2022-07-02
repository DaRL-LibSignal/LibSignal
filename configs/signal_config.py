# notes: the order in one list is unorder
# for example: in cityflow, it maybe 'WT_ET', but in sumo, it maybe 'ET_WT'
# the order in those two lists is different, just ignore it.
signal_configs = {
    'grid4x4': {
        'phase_pairs': [[1, 7], [2, 8], [1, 2], [7, 8], [4, 10], [5, 11], [10, 11], [4, 5]],
        'valid_acts': None
    },
    'hz1x1': {
        # 'sumo_phase_pairs': [[2, 6], [0, 4], [3, 7], [1, 5], [6, 7], [2, 3], [4, 5], [0, 1]],
        # 'sumo_valid_acts': None,
        'phase_pairs': [[0, 4], [2, 7], [1, 5], [3, 6], [0, 1], [4, 5], [2, 3], [6, 7]],
        'valid_acts': None
    },
    'hz1x1_config2': {
        # 'sumo_phase_pairs': [],
        # 'sumo_valid_acts': None,
		# TODO check whether is correct, can we merge it into hz1x1?
        'phase_pairs': [[0, 4], [2, 7], [1, 5], [3, 6], [0, 1], [4, 5], [2, 3], [6, 7]],
        'valid_acts': None
    },
    'hz1x1_config3': {
        # 'sumo_phase_pairs': [],
        # 'sumo_valid_acts': None,
		# TODO check whether is correct, can we merge it into hz1x1?
        'phase_pairs': [[0, 4], [2, 7], [1, 5], [3, 6], [0, 1], [4, 5], [2, 3], [6, 7]],
        'valid_acts': None
    },
    'hz1x1_config4': {
        # 'sumo_phase_pairs': [],
        # 'sumo_valid_acts': None,
		# TODO check whether is correct, can we merge it into hz1x1?
        'phase_pairs': [[0, 4], [2, 7], [1, 5], [3, 6], [0, 1], [4, 5], [2, 3], [6, 7]],
        'valid_acts': None
    },
    'hz4x4': {
        # 'sumo_phase_pairs': [],
        # 'sumo_valid_acts': None,
        'phase_pairs': [],
        'valid_acts': None
    },
    'cologne1': {
        'phase_pairs': [[1, 7], [2, 8], [4, 10], [5, 11]],
        'valid_acts': None,
        # 'cityflow_phase_pairs': [],
        # 'cityflow_valid_acts': None
    },
    'cologne3': {
        'phase_pairs': [[1, 7], [2, 8], [1, 2], [7, 8], [4, 10], [5, 11], [10, 11], [4, 5], [9, 11]],
        'valid_acts': {
            'GS_cluster_2415878664_254486231_359566_359576': {4: 0, 5: 1, 0: 2, 1: 3},
            '360086': {4: 0, 5: 1, 0: 2, 1: 3},
            '360082': {4: 0, 5: 1, 1: 2},
        },
        # 'cityflow_phase_pairs': [],
        # 'cityflow_valid_acts': {
        #     'GS_cluster_2415878664_254486231_359566_359576': {},
        #     '360086': {},
        #     '360082': {}
        # }
    }

}
