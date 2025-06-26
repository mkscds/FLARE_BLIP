from typing import List, Optional, Tuple

import torch
import numpy as np
import random

from torch.utils.data import DataLoader, random_split, TensorDataset

from .dataset_preparation_fedmd import (FEDMD_partition_data, FEDMD_partition_data_dirichlet, FEDMD_partition_data_label_quantity)
from Models.models import BLIPVisionWrapper
from Config import parse_arguments

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

args = parse_arguments()
set_seed(seed=args.seed)


def FEDMD_load_datasets_noblip(args):
    train_datasets = []
    val_datasets = []

    partitioning = args.partitioning

    if partitioning == "dirichlet":
        datasets, testset, serverset = FEDMD_partition_data_dirichlet(args.num_clients, alpha=args.alpha, seed=args.seed, dataset_name=args.name)
    
    elif partitioning == "label_quantity":
        datasets, testset, serverset = FEDMD_partition_data_label_quantity(args.num_clients, labels_per_client=args.labels_per_client, seed=args.seed, dataset_name=args.name)

    elif partitioning == "iid":
        datasets, testset, serverset = FEDMD_partition_data(args.num_clients, similarity=1.0, seed=args.seed, dataset_name=args.name)

    elif partitioning == "iid_noniid":
        datasets, testset, serverset = FEDMD_partition_data(args.num_clients, similarity=args.similarity, seed=args.seed, dataset_name=args.name)
    
    for dataset in datasets:
        len_val = int(len(dataset)*args.val_ratio) if args.val_ratio > 0 else 0
        lengths = [len(dataset) - len_val, len_val]
        ds_train, ds_val = random_split(dataset, lengths, torch.Generator().manual_seed(args.seed))
        train_datasets.append(ds_train)
        val_datasets.append(ds_val)

    return train_datasets, val_datasets, testset, serverset

def FEDMD_load_dataloaders_noblip(args):
    trainloaders, valloaders = [], []
    train_datasets, val_datasets, testset, serverset = FEDMD_load_datasets_noblip(args)
    batch_size = args.batch_size
    for i, (trainset, valset) in enumerate(zip(train_datasets, val_datasets)):
        trainloaders.append(DataLoader(trainset, batch_size=batch_size, shuffle=True))
        valloaders.append(DataLoader(valset, batch_size=batch_size, shuffle=False))

    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    serverloader = DataLoader(serverset, batch_size=batch_size, shuffle=False)

    return trainloaders, valloaders, testloader, serverloader


def FEDMD_load_dataloaders_blip(args):
    blip_wrappers = BLIPVisionWrapper()
    trainloaders_blip, valloaders_blip = [], []
    trainsets, valsets, testset, serverset = FEDMD_load_datasets_noblip(args)
    for i, (trainset, valset) in enumerate(zip(trainsets, valsets)):
        _, train_embeddings, train_labels = blip_wrappers.process_dataset(trainset, cache_file=f'dataset/{args.name}/trainset_{args.alpha}/train_blip_embeddings_{i}.npz')
        _, val_embeddings, val_labels = blip_wrappers.process_dataset(valset, cache_file=f'dataset/{args.name}/valset_{args.alpha}/val_blip_embeddings_{i}.npz')

        ds_train = TensorDataset(torch.tensor(train_embeddings), torch.tensor(train_labels))
        ds_val = TensorDataset(torch.tensor(val_embeddings), torch.tensor(val_labels))

        train_loader = DataLoader(ds_train, batch_size = args.batch_size, shuffle=True)
        val_loader = DataLoader(ds_val, batch_size = args.batch_size, shuffle=False)
        trainloaders_blip.append(train_loader)
        valloaders_blip.append(val_loader)

    _, test_embeddings, test_labels = blip_wrappers.process_dataset(testset, cache_file=f'dataset/{args.name}/test_blip_embeddings.npz')
    ds_test = TensorDataset(torch.tensor(test_embeddings), torch.tensor(test_labels))
    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False)
    _, server_embeddings, server_labels = blip_wrappers.process_dataset(serverset, cache_file=f'dataset/{args.name}/server_blip_embeddings.npz')
    ds_server = TensorDataset(torch.tensor(server_embeddings), torch.tensor(server_labels))
    server_loader = DataLoader(ds_server, batch_size=args.batch_size, shuffle=False)
    return trainloaders_blip, valloaders_blip, test_loader, server_loader

if __name__=="__main__":
    args = parse_arguments()
    train_datasets, val_datasets, testset, serverset = FEDMD_load_datasets_noblip(args)

