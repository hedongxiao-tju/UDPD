import numpy as np
import torch
import random
import os

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T


def get_cora_casestudy(SEED=0):
    data_X, data_Y, data_citeid, data_edges = parse_cora()

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    data_name = "cora"
    dataset = Planetoid("dataset", data_name, transform=T.NormalizeFeatures())
    data = dataset[0]

    data.x = torch.tensor(data_X).float()
    data.edge_index = torch.tensor(data_edges).long()
    data.y = torch.tensor(data_Y).long()
    data.num_nodes = len(data_Y)

    node_id = np.arange(data.num_nodes)
    np.random.shuffle(node_id)

    data.train_id = np.sort(node_id[: int(data.num_nodes * 0.60)])
    data.val_id = np.sort(node_id[int(data.num_nodes * 0.60) : int(data.num_nodes * 0.80)])
    data.test_id = np.sort(node_id[int(data.num_nodes * 0.80) :])

    data.train_mask = torch.tensor([x in data.train_id for x in range(data.num_nodes)])
    data.val_mask = torch.tensor([x in data.val_id for x in range(data.num_nodes)])
    data.test_mask = torch.tensor([x in data.test_id for x in range(data.num_nodes)])

    return data, data_citeid


def parse_cora():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "dataset", "cora_orig", "cora")

    idx_features_labels = np.genfromtxt(f"{path}.content", dtype=np.dtype(str))
    data_X = idx_features_labels[:, 1:-1].astype(np.float32)
    labels = idx_features_labels[:, -1]
    class_map = {
        x: i
        for i, x in enumerate(
            [
                "Case_Based",
                "Genetic_Algorithms",
                "Neural_Networks",
                "Probabilistic_Methods",
                "Reinforcement_Learning",
                "Rule_Learning",
                "Theory",
            ]
        )
    }
    data_Y = np.array([class_map[l] for l in labels])
    data_citeid = idx_features_labels[:, 0]
    idx = np.array(data_citeid, dtype=np.dtype(str))
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(f"{path}.cites", dtype=np.dtype(str))
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(edges_unordered.shape)
    data_edges = np.array(edges[~(edges == None).max(1)], dtype="int")
    data_edges = np.vstack((data_edges, np.fliplr(data_edges)))
    return data_X, data_Y, data_citeid, np.unique(data_edges, axis=0).transpose()


def get_raw_text_cora(use_text=False, seed=0):
    data, data_citeid = get_cora_casestudy(seed)
    if not use_text:
        return data, None

    base_dir = os.path.dirname(os.path.abspath(__file__))
    papers_path = os.path.join(base_dir, "dataset", "cora_orig", "mccallum", "cora", "papers")

    with open(papers_path) as f:
        lines = f.readlines()
    pid_filename = {}
    for line in lines:
        pid = line.split("\t")[0]
        fn = line.split("\t")[1]
        fn = fn.replace(":", "_")
        if fn == "http_##www.cs.ucc.ie#~dgb#papers#ICCBR2.ps.Z":
            fn = "http_##www.cs.ucc.ie#~dgb#papers#iccbr2.ps.Z"
        if fn == "http_##www.cs.ucl.ac.uk#staff#t.yu#pgp.new.ps":
            fn = "http_##www.cs.ucl.ac.uk#staff#T.Yu#pgp.new.ps"
        pid_filename[pid] = fn

    extractions_path = os.path.join(base_dir, "dataset", "cora_orig", "mccallum", "cora", "extractions")

    text = []
    missing_files = 0
    missing_details = []

    for pid in data_citeid:
        fn = pid_filename[pid]
        ti = "Title: Unknown"
        ab = "Abstract: No abstract available"

        try:
            with open(os.path.join(extractions_path, fn)) as f:
                lines = f.read().splitlines()

            for line in lines:
                if "Title:" in line:
                    ti = line
                if "Abstract:" in line:
                    ab = line
        except FileNotFoundError:
            missing_files += 1
            missing_details.append((pid, fn))

        text.append(ti + "\n" + ab)

    if missing_files > 0:
        print(f"Warning: {missing_files} text files are missing; using default text instead.")
        print("Missing file details:")
        for pid, fn in missing_details:
            full_path = os.path.join(extractions_path, fn)
            print(f"  - Paper ID: {pid}")
            print(f"    Filename: {fn}")
            print(f"    Full path: {full_path}")

    return data, text
