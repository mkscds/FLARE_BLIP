"""Download data and partition data with different partitioning strategies."""
import os, random
from typing import List, Tuple
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import ConcatDataset, Dataset, Subset
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST
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


def plot_and_save_class_distribution(trainsets_per_client, trainset_after_server, num_clients, num_classes, alpha, plot_type='table'):
    """
    Visualize and save class distribution across clients.
    
    Parameters:
    - trainsets_per_client: List of client datasets
    - trainset_after_server: The training set after server split
    - num_clients: Number of clients
    - num_classes: Number of classes in dataset
    - alpha: Dirichlet alpha parameter (for title)
    - plot_type: 'bar' for bar chart or 'table' for table visualization
    """
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Gather data
    client_data = []
    for client_id, client_dataset in enumerate(trainsets_per_client):
        targets = np.array(trainset_after_server.dataset.targets)[client_dataset.indices]
        unique, counts = np.unique(targets, return_counts=True)
        for class_id, count in zip(unique, counts):
            client_data.append({
                'Client': f'Client {client_id}',
                'Class': class_id,
                'Count': count
            })
    
    df = pd.DataFrame(client_data)
    
    if plot_type == 'bar':
        # Create bar plot
        plt.figure(figsize=(12, 6))
        for class_id in range(num_classes):
            class_data = df[df['Class'] == class_id]
            plt.bar(class_data['Client'], class_data['Count'], label=f'Class {class_id}', alpha=0.7)
        
        plt.title(f'Class Distribution Across Clients (Dirichlet α={alpha})')
        plt.xlabel('Client ID')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45)
        plt.legend(title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'plots/dirichlet_distribution_alpha_{alpha}.png', bbox_inches='tight')
        plt.close()
        
    elif plot_type == 'table':
        # Create table visualization
        plt.figure(figsize=(12, 8))
        
        # Create a matrix of counts
        distribution_matrix = np.zeros((num_clients, num_classes))
        for client_id, client_dataset in enumerate(trainsets_per_client):
            targets = np.array(trainset_after_server.dataset.targets)[client_dataset.indices]
            unique, counts = np.unique(targets, return_counts=True)
            for class_id, count in zip(unique, counts):
                distribution_matrix[client_id, class_id] = count
        
        # Create table
        fig, ax = plt.subplots()
        ax.axis('off')
        ax.axis('tight')
        
        # Create column and row labels
        columns = [f'Class {i}' for i in range(num_classes)]
        rows = [f'Client {i}' for i in range(num_clients)]
        
        # Create table
        table = ax.table(cellText=distribution_matrix.astype(int),
                        rowLabels=rows,
                        colLabels=columns,
                        cellLoc='center',
                        loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
        
        plt.title(f'Class Distribution Across Clients (Dirichlet α={alpha})')
        plt.savefig(f'plots/dirichlet_distribution_table_alpha_{alpha}.png', bbox_inches='tight')
        plt.close()

def _download_data(dataset_name="emnist") -> Tuple[Dataset, Dataset]:
    """Download the requested dataset. Currently supports cifar10, mnist, and fmniost.

    Returns:
    ----------
    Tuple[Dataset, testset] 
        The training dataset, the test data
    
    """
    trainset, testset = None, None

    if dataset_name == "cifar10":
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(
                    lambda x: F.pad(
                        Variable(x.unsqueeze(0), requires_grad = False),
                        (4, 4, 4, 4),
                        mode="reflect",
                    ).data.squeeze()
                ),
                transforms.ToPILImage(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])

        trainset = CIFAR10(root='./data/data_cifar10', train=True, download=True, transform=transform_train)
        testset = CIFAR10(root='./data/data_cifar10', train=False, download=True, transform=transform_test)
    
    elif dataset_name == "mnist":
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        trainset = MNIST(root='./data/data_mnist', train=True, download=True, transform=transform_train)
        testset = MNIST(root='./data/data_mnist', train=False, download=True, transform=transform_test)

    elif dataset_name == "fmnist":
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        trainset = FashionMNIST(root='./data/data_fmnist', train=True, download=True, transform=transform_train)
        testset = FashionMNIST(root='./data/data_fmnist', train=False, download=True, transform=transform_test)


    elif dataset_name == "cifar100":
        transform_train = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Lambda( lambda x: F.pad( Variable(x.unsqueeze(0), requires_grad=False), (4, 4, 4, 4), mode="reflect", ).data.squeeze()),
                        transforms.ToTensor(),
                        transforms.ToPILImage(),
                        transforms.RandomCrop(32),
                        transforms.RandomHorizontalFlip(),
                            ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        trainset = CIFAR100(root='./data/data_cifar100', train=True, download=True, transform=transform_train)
        testset = CIFAR100(root='./data/data_cifar100', train=False, download=True, transform=transform_test)

    else:
        raise NotImplementedError
    
    return trainset, testset



def partition_data(num_clients, similarity=1.0, seed=42, dataset_name="cifar10") -> Tuple[List[Dataset], Dataset]:
    """Partition the dataset into subsets for each client.
    
    Parameters:
    -----------

    num_clients: int
        The number of clients that hold a part of the data

    similarity: float
        Parameter to sample similar data

    seed: int, optional
        Used to set a fix seed to replicate experiments, by default 42

    Returns:
    --------
    Tuple[List[Subset], Dataset]
        The list of datasets for each client, the test dataset    
    
    """
    trainset, testset = _download_data(dataset_name)
    trainsets_per_client = []
    # for s% similarity sample iid data per client
    # Step 1: Separate 10% for serverset
    total_samples = len(trainset)
    server_fraction = int(0.1 * total_samples)  # 10% of trainset

    prng = np.random.default_rng(seed)
    server_idxs = prng.choice(total_samples, server_fraction, replace=False)

    serverset = Subset(trainset, server_idxs)
    trainset_after_server = Subset(trainset, np.setdiff1d(np.arange(total_samples), server_idxs))

    # Step 2: Divide remaining 90% into iid_trainset and rem_trainset
    remaining_samples = len(trainset_after_server)
    s_fraction = int(similarity * remaining_samples)  # Similarity fraction of remaining data

    idxs = prng.choice(remaining_samples, s_fraction, replace=False)
    iid_trainset = Subset(trainset_after_server, idxs)
    rem_trainset = Subset(trainset_after_server, np.setdiff1d(np.arange(remaining_samples), idxs))
    
    
    # Sample iid data per client from iid_trainset
    all_ids = np.arange(len(iid_trainset))
    splits = np.array_split(all_ids, num_clients)
    for i in range(num_clients):
        c_ids = splits[i]
        d_ids = iid_trainset.indices[c_ids]
        trainsets_per_client.append(Subset(iid_trainset.dataset, d_ids))

    if similarity == 1.0:
        return trainsets_per_client, testset, serverset
    
    tmp_t = rem_trainset.dataset.targets
    if isinstance(tmp_t, list):
        tmp_t = np.array(tmp_t)
    if isinstance(tmp_t, torch.Tensor):
        tmp_t = tmp_t.numpy()

    targets = tmp_t[rem_trainset.indices]
    num_remaining_classes = len(set(targets))
    remaining_classes = list(set(targets))
    client_classes: List[List] = [[] for _ in range(num_clients)]
    times = [0 for _ in range(num_remaining_classes)]

    for i in range(num_clients):
        client_classes[i] = [remaining_classes[i % num_remaining_classes]]
        times[i % num_remaining_classes] += 1
        j = 1
        while j < 2:
            index = prng.choice(num_remaining_classes)
            class_t = remaining_classes[index]
            if class_t not in client_classes[i]:
                client_classes[i].append(class_t)
                times[index] += 1
                j += 1

    
    rem_trainsets_per_client: List[List] = [[] for _ in range(num_clients)]

    for i in range(num_remaining_classes):
        class_t = remaining_classes[i]
        idx_k = np.where(targets == i)[0]
        prng.shuffle(idx_k)
        idx_k_split = np.array_split(idx_k, times[i])
        ids = 0
        for j in range(num_clients):
            if class_t in client_classes[j]:
                act_idx = rem_trainset.indices[idx_k_split[ids]]
                rem_trainsets_per_client[j].append(Subset(rem_trainset, act_idx))
                ids+= 1

    
    for i in range(num_clients):
        trainsets_per_client[i] = ConcatDataset([trainsets_per_client[i]] + rem_trainsets_per_client[i])

    
    return trainsets_per_client, testset, serverset


def partition_data_dirichlet(num_clients, alpha, seed=42, dataset_name="cifar10") -> Tuple[List[Dataset], Dataset]:
    """Partition according to the Dirichlet distribution.
    
    Parameters
    ----------
    num_clients: int
        The number of clients that hold a part of the data

    alpha: float
        Parameter of the Dirichlet distribution

    seed: int, optional
        Used to set a fix seed to replicate experiment by default 42

    dataset_name: str
        Name of the dataset to be used
    
    Returns
    -------
    Tuple[List[Subset], Dataset]
        The list of datasets for each client, the test dataset

    """
    trainset, testset = _download_data(dataset_name)
    min_samples = 0
    min_required_samples_per_client = 10
    trainsets_per_client = []
    # for s% similarity sample iid data per client
    # Step 1: Separate 10% for serverset
    total_samples = len(trainset)
    server_fraction = int(0.1 * total_samples)  # 10% of trainset

    prng = np.random.default_rng(seed)
    server_idxs = prng.choice(total_samples, server_fraction, replace=False)

    serverset = Subset(trainset, server_idxs)

    trainset_after_server = Subset(trainset, np.setdiff1d(np.arange(total_samples), server_idxs))
    tmp_t = np.array(trainset_after_server.dataset.targets)[trainset_after_server.indices]
    print(len(tmp_t))
    if isinstance(tmp_t, list):
        tmp_t = np.array(tmp_t)
    if isinstance(tmp_t, torch.Tensor):
        tmp_t = tmp_t.numpy()

    num_classes = len(set(tmp_t))
    total_samples = len(tmp_t)

    while min_samples < min_required_samples_per_client:
        idx_clients: List[List] = [[] for _ in range(num_clients)]
        for k in range(num_classes):
            idx_k = np.where(tmp_t == k)[0]
            prng.shuffle(idx_k)
            proportions = prng.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array(
                [
                    p * (len(idx_j) < total_samples / num_clients) for p, idx_j in zip(proportions, idx_clients)
                ]
            )
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_k_split = np.split(idx_k, proportions)
            idx_clients = [
                idx_j + idx.tolist() for idx_j, idx in zip(idx_clients, idx_k_split)
            ]
            min_samples = min([len(idx_j) for idx_j in idx_clients])

    trainsets_per_client = [Subset(trainset_after_server, idxs) for idxs in idx_clients]
    
    plot_and_save_class_distribution(
        trainsets_per_client=trainsets_per_client,
        trainset_after_server=trainset_after_server,
        num_clients=num_clients,
        num_classes=num_classes,
        alpha=alpha,
        plot_type='table'  # or 'table' for table visualization
    )
    return trainsets_per_client, testset, serverset

def partition_data_label_quantity(num_clients, labels_per_client, seed=42, dataset_name="cifar10") -> Tuple[List[Dataset], Dataset]:
    """Partition the data according to the number of labels per client.
    
    Parameters:
    ----------
    num_clients: int
        The number of clients that hold a part of the data

    num_labels_per_client: int
        NUmber of labels per clients

    seed: int, optional
        Used to set a fix seed to replicate experiments, by default 42

    dataset_name: str
        Name of the dataset to be used.    
    
    Return:
    Tuple[List[Subset], Dataset]    
        The list of the datasets for each client, the test dataset.
    """
    
    trainset, testset = _download_data(dataset_name)
    prng = np.random.default_rng(seed)

    total_samples = len(trainset)
    server_fraction = int(0.1 * total_samples)  # 10% of trainset

    prng = np.random.default_rng(seed)
    server_idxs = prng.choice(total_samples, server_fraction, replace=False)

    serverset = Subset(trainset, server_idxs)
    trainset_after_server = Subset(trainset, np.setdiff1d(np.arange(total_samples), server_idxs))

    targets = trainset_after_server.dataset.targets

    if isinstance(targets, list):
        targets = np.array(targets)
    
    if isinstance(targets, torch.Tensor):
        targets = targets.numpy()

    num_classes = len(set(targets))
    times = [0 for _ in range(num_classes)]
    contains = []

    for i in range(num_clients):
        current = [ i % num_classes]
        times[i % num_classes] += 1
        j = 1
        while j < labels_per_client:
            index = prng.choice(num_classes, 1)[0]
            if index not in current:
                current.append(index)
                times[index] += 1
        
        contains.append(current)

    idx_clients: List[List] = [[] for _ in range(num_clients)]
    for i in range(num_classes):
        idx_k = np.where(targets == i)[0]
        prng.shuffle(idx_k)
        idx_k_split = np.array_split(idx_k, times[i])
        ids = 0
        for j in range(num_clients):
            if i in contains[j]:
                idx_clients[j] += idx_k_split[ids].tolist()
                ids += 1

    trainset_after_client = [Subset(trainset_after_server, idxs) for idxs in idx_clients]
    
    return trainset_after_client, testset, serverset













