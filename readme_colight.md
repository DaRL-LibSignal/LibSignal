light 用法说明
命令参考如下：（命令行中需要调整的参数基本都在下面了）
python3 run_colight.py --config_file jn343 --graph_info_dir jn34 -lr 1e-3 -bs 64 -ls 1000 -ep 0.8 -ed 0.9995 -me 0.01 -rs 5000 --episodes 100 --action_interval 10 -pr yzy27 --ngpu 0
    config_file:指定cityflow的config，详细路径为 data/config_dir/config_{config_file}.json
    graph_info_dir:存储了路网的邻接矩阵一类的信息。 详细路径为 data/graphinfo/graph_info_{graph_info_dir}.pkl  可以利用utils.py中的build_int_intersection_map（）生成相关pkl文件

对于colight中网络维度，multi-head数目，都可以在run_colight.py中的dic_graph_setting进行调整
大家可以将相关文件复制到自己的workspace进行尝试，主要是environment.py（我对environmnet.py做了一些修改，故而和tlc-baseline中的有些区别），run_colight.py 以及 colight_agtnt.py 以及utils.py.
具体的文件层次信息主要仿照我data中结构即可