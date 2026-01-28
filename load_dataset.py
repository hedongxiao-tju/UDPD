import numpy as np
import torch
import random
import os

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T


# return cora dataset as pytorch geometric Data object together with 60/20/20 split, and list of cora IDs


def get_cora_casestudy(SEED=0):
    data_X, data_Y, data_citeid, data_edges = parse_cora()
    # data_X = sklearn.preprocessing.normalize(data_X, norm="l1")

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)  # Numpy module.
    random.seed(SEED)  # Python random module.

    # load data
    data_name = 'cora'
    # path = osp.join(osp.dirname(osp.realpath(__file__)), 'dataset')
    dataset = Planetoid('dataset', data_name,
                        transform=T.NormalizeFeatures())
    data = dataset[0]

    data.x = torch.tensor(data_X).float()
    data.edge_index = torch.tensor(data_edges).long()
    data.y = torch.tensor(data_Y).long()
    data.num_nodes = len(data_Y)

    # split data - 平衡划分: 70/20/10 (平衡训练数据和验证稳定性)
    node_id = np.arange(data.num_nodes)
    np.random.shuffle(node_id)

    data.train_id = np.sort(node_id[:int(data.num_nodes * 0.70)])
    data.val_id = np.sort(
        node_id[int(data.num_nodes * 0.70):int(data.num_nodes * 0.90)])
    data.test_id = np.sort(node_id[int(data.num_nodes * 0.90):])

    data.train_mask = torch.tensor(
        [x in data.train_id for x in range(data.num_nodes)])
    data.val_mask = torch.tensor(
        [x in data.val_id for x in range(data.num_nodes)])
    data.test_mask = torch.tensor(
        [x in data.test_id for x in range(data.num_nodes)])

    return data, data_citeid

# credit: https://github.com/tkipf/pygcn/issues/27, xuhaiyun


def parse_cora():
    # === 路径配置 ===
    # Windows/Linux通用: 使用相对路径（推荐）
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, 'dataset', 'cora_orig', 'cora')
    
    # Linux环境使用（需要时取消注释并注释掉上面两行）
    # path = '/mnt/lun1/home/jd/code/ljc/dual-diffusion-graph-model/dataset/cora_orig/cora'
    
    idx_features_labels = np.genfromtxt(
        "{}.content".format(path), dtype=np.dtype(str))
    data_X = idx_features_labels[:, 1:-1].astype(np.float32)
    labels = idx_features_labels[:, -1]
    class_map = {x: i for i, x in enumerate(['Case_Based', 'Genetic_Algorithms', 'Neural_Networks',
                                            'Probabilistic_Methods', 'Reinforcement_Learning', 'Rule_Learning', 'Theory'])}
    data_Y = np.array([class_map[l] for l in labels])
    data_citeid = idx_features_labels[:, 0]
    idx = np.array(data_citeid, dtype=np.dtype(str))
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(
        "{}.cites".format(path), dtype=np.dtype(str))
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(
        edges_unordered.shape)
    data_edges = np.array(edges[~(edges == None).max(1)], dtype='int')
    data_edges = np.vstack((data_edges, np.fliplr(data_edges)))
    return data_X, data_Y, data_citeid, np.unique(data_edges, axis=0).transpose()


def get_raw_text_cora(use_text=False, seed=0):
    data, data_citeid = get_cora_casestudy(seed)
    if not use_text:
        return data, None

    # === 路径配置 ===
    # Windows/Linux通用: 使用相对路径（推荐）
    base_dir = os.path.dirname(os.path.abspath(__file__))
    papers_path = os.path.join(base_dir, 'dataset', 'cora_orig', 'mccallum', 'cora', 'papers')
    
    # Linux环境使用（需要时取消注释并注释掉上面两行）
    # papers_path = '/mnt/lun1/home/jd/code/ljc/dual-diffusion-graph-model/dataset/cora_orig/mccallum/cora/papers'
    
    with open(papers_path) as f:
        lines = f.readlines()
    pid_filename = {}
    for line in lines:
        pid = line.split('\t')[0]
        fn = line.split('\t')[1]
        fn = fn.replace(':', '_')
        if fn == 'http_##www.cs.ucc.ie#~dgb#papers#ICCBR2.ps.Z':
            fn = 'http_##www.cs.ucc.ie#~dgb#papers#iccbr2.ps.Z'
        if fn == 'http_##www.cs.ucl.ac.uk#staff#t.yu#pgp.new.ps':
            fn = 'http_##www.cs.ucl.ac.uk#staff#T.Yu#pgp.new.ps'
        pid_filename[pid] = fn

    # === 路径配置 ===
    # Windows/Linux通用: 使用相对路径（推荐）
    extractions_path = os.path.join(base_dir, 'dataset', 'cora_orig', 'mccallum', 'cora', 'extractions')
    
    # Linux环境使用（需要时取消注释并注释掉上面一行）
    # extractions_path = '/mnt/lun1/home/jd/code/ljc/dual-diffusion-graph-model/dataset/cora_orig/mccallum/cora/extractions'
    
    text = []
    missing_files = 0
    missing_details = []
    
    for pid in data_citeid:
        fn = pid_filename[pid]
        ti = 'Title: Unknown'
        ab = 'Abstract: No abstract available'
        
        try:
            with open(os.path.join(extractions_path, fn)) as f:
                lines = f.read().splitlines()
            
            for line in lines:
                if 'Title:' in line:
                    ti = line
                if 'Abstract:' in line:
                    ab = line
        except FileNotFoundError:
            missing_files += 1
            missing_details.append((pid, fn))
            # 使用默认文本，不影响训练
        
        text.append(ti+'\n'+ab)
    
    if missing_files > 0:
        print(f"警告: {missing_files} 个文本文件缺失，使用默认文本代替")
        print(f"缺失的文件详情:")
        for pid, fn in missing_details:
            full_path = os.path.join(extractions_path, fn)
            print(f"  - Paper ID: {pid}")
            print(f"    文件名: {fn}")
            print(f"    完整路径: {full_path}")
    
    return data, text