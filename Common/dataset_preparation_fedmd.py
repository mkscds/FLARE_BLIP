from typing import List, Tuple
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import ConcatDataset, Dataset, Subset
from torchvision.datasets import CIFAR10, MNIST, CIFAR100
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


def _download_data(dataset_name="cifar100"):
    """Download the requsested dataset.
    Returns
    _______

    Tuple[Dataset, Dataset]
        The training datset, the test data
    """
    trainset, testset = None, None

    if dataset_name == "cifar10":
        transform_train = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Lambda(lambda x: F.pad(Variable(x.unsqueeze(0), requires_grad=False), (4, 4, 4, 4), mode="reflect").data.squeeze()),
                            transforms.RandomCrop(32),
                            transforms.RandomHorizontalFlip(),])
        
        transform_test = transforms.Compose([transforms.ToTensor(),])

        trainset = CIFAR10(root='./data/data_cifar10', train=True, download=True, transform=transform_train)
        testset = CIFAR10(root='./data/data_cifar10', train=False, download=True, transform=transform_test)

    elif dataset_name == "cifar100":
        transform_train = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Lambda(lambda x: F.pad(Variable(x.unsqueeze(0), requires_grad=False), (4, 4, 4, 4), mode="reflect").data.squeeze()),
                            transforms.RandomCrop(32),
                            transforms.RandomHorizontalFlip(),])
        
        #transforms.ToPILImage(),

        transform_test = transforms.Compose([transforms.ToTensor(),])

        trainset = CIFAR100(root='./data/data_cifar100', train=True, download=True, transform=transform_train)
        testset = CIFAR100(root='./data/data_cifar100', train=False, download=True, transform=transform_test)

    elif dataset_name == "mnist":
        transform_test = transforms.Compose([transforms.ToTensor(),])
        transform_train = transforms.Compose([transforms.ToTensor(),])

        trainset = MNIST(root='./data/data_mnist', train=True, download=True, transform=transform_train)
        testset = MNIST(root='./data/data_mnist', train=False, download=True, transform=transform_test)


    return trainset, testset


def FEDMD_partition_data(num_clients, similarity=1.0, seed=42, dataset_name="cifar100", server_data_fraction=0.1):

    prng = np.random.default_rng(seed)
    trainsets_per_client = []
    trainset, testset = _download_data(dataset_name)

    if dataset_name == "cifar100":
        serverset_full, _ = _download_data(dataset_name="cifar10")
        total_len = len(serverset_full)
        subset_len = int(total_len * server_data_fraction)
        idxs = prng.choice(total_len, subset_len, replace=False)
        serverset = Subset(serverset_full, idxs)

    remaining_samples = len(trainset)
    s_fraction = int(similarity * s_fraction)
    idxs = prng.choice(remaining_samples, s_fraction, replace=False)

    iid_trainset = Subset(trainset, idxs)
    rem_trainset = Subset(trainset, np.setdiff1d(np.arange(remaining_samples), idxs))

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
        tmp_t =  np.array(tmp_t)

    if isinstance(tmp_t, torch.tensor):
        tmp_t = tmp_t.numpy()

    
    targets = tmp_t[rem_trainset.indices]

    num_remaining_classes = len(set(targets))
    remaining_classes = list(set(targets))
    client_classes: list[list] = [[] for _ in range(num_clients)]
    times = [0 for _ in range(num_remaining_classes)]

    for i in range(num_clients):
        client_classes[i] = [remaining_classes[i % num_remaining_classes]]
        times[i% num_remaining_classes] += 1
        j = 1
        while j < 2:
            index = prng.choice(num_remaining_classes)
            class_t = remaining_classes[index]
            if class_t not in client_classes[i]:
                client_classes[i].append(class_t)
                times[index] += 1
                j += 1

    rem_trainsets_per_client = [[] for _ in range(num_clients)]

    for i in range(num_remaining_classes):
        class_t = remaining_classes[i]
        idx_k = np.where(targets == i)
        prng.shuffle(idx_k)
        idx_k_split = np.array_split(idx_k, times[i])
        ids = 0
        for j in range(num_clients):
            if class_t in client_classes[j]:
                act_idx = rem_trainset.indices[idx_k_split[ids]]
                rem_trainsets_per_client[j].append(Subset(rem_trainset, act_idx))


    for i in range(num_clients):
        trainsets_per_client[i] = ConcatDataset([trainsets_per_client[i]] + rem_trainsets_per_client[i])


    return trainsets_per_client, testset, serverset

def FEDMD_partition_data_dirichlet(num_clients, alpha, seed=42, dataset_name="cifar10", server_data_fraction=0.1):
    """Partition according to the Dirichlet distribution"""

    prng = np.random.default_rng(seed)
    trainset, testset = _download_data(dataset_name)
    if dataset_name == 'cifar100':
        serverset_full, _ = _download_data(dataset_name='cifar10')
        total_len = len(serverset_full)
        subset_len = int(server_data_fraction * total_len)
        idxs = prng.choice(total_len, subset_len, replace=False)
        serverset = Subset(serverset_full, idxs)

    elif dataset_name == 'femnist':
        pass


    min_samples = 0
    min_required_samples_per_client = 10
    trainsets_per_client = []
    total_samples = len(trainset)
    tmp_t = trainset.targets
    if isinstance(tmp_t, list):
        tmp_t = np.array(tmp_t)
    if isinstance(tmp_t, torch.Tensor):
        tmp_t = tmp_t.numpy()

    num_classes = len(set(tmp_t))
    total_samples = len(tmp_t)
    while min_samples < min_required_samples_per_client:
        idx_clients = [[] for _ in range(num_clients)]
        for k in range(num_classes):
            idx_k = np.where(tmp_t == k)[0]
            prng.shuffle(idx_k)
            proportions = prng.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array([p*(len(idx_j) < total_samples / num_clients) for p, idx_j in zip(proportions, idx_clients)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_k_split = np.split(idx_k, proportions)
            idx_clients = [ idx_j + idx.tolist() for idx_j, idx in zip(idx_clients, idx_k_split)]
            min_samples = min([len(idx_j) for idx_j in idx_clients])
    
    trainsets_per_client = [Subset(trainset, idxs) for idxs in idx_clients]

    return trainsets_per_client, testset, serverset

def FEDMD_partition_data_label_quantity(num_clients, labels_per_client, seed=42, dataset_name="cifar100", server_data_fraction=0.1):
    """Partition the data according to the number of labels per clients"""

    prng = np.random.default_rng(seed)
    trainset, testset = _download_data(dataset_name)
    if dataset_name == "cifar100":
        serverset_full, _ = _download_data(dataset_name="cifar10")
        total_len = len(serverset_full)
        subset_len = int(server_data_fraction*total_len)
        idxs = prng.choice(total_len, subset_len, replace=False)
        serverset = Subset(serverset_full, idxs)
    
    targets = trainset.targets
    if isinstance(targets, list):
        targets = np.array(targets)
    if isinstance(targets, torch.Tensor):
        targets = targets.numpy()

    num_classes = len(set(targets))
    times = [0 for _ in range(num_classes)]
    contains = []

    for i in range(num_clients):
        current = [i % num_classes]
        times[i % num_classes] += 1
        j = 1
        while j < labels_per_client:
            index = prng.choice(num_classes, 1)[0]
            if index not in current:
                current.append(index)
                times[index] += 1
                j += 1
        contains.append(current)

    idx_clients = [[] for _ in range(num_clients)]

    for i in range(num_classes):
        idx_k = np.where(targets == i)[0]
        prng.shuffle(idx_k)
        idx_k_split = np.array_split(idx_k, times[i])
        ids = 0
        for j in range(num_clients):
            idx_clients[j] += idx_k_split[ids].tolist()
            ids += 1
        
    trainsets_per_client = [Subset(trainset, idxs) for idxs in idx_clients]
    return trainsets_per_client, testset, serverset
