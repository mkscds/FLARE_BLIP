import sys
sys.modules['tkinter'] = None
import os
import pickle
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset

import flwr as fl

from typing import Dict, List, Tuple, Optional
from Config import parse_arguments
from Common.dataset_fedmd import FEDMD_load_dataloaders_blip, FEDMD_load_dataloaders_noblip
from Common.dataset import load_dataloaders_blip, load_dataloaders_noblip
from Models.model_utils import test, set_parameters
from Models.models import NetS, MLP4
from Clients.client_fedavg import gen_client_fn_NOKD


from flwr.server.client_manager import SimpleClientManager
from flwr.server.server import Server
from flwr.server.strategy import FedAvg
from Strategies.strategy_fedprox import FedProx


def run(args):

    server_side_loss = []
    server_side_accuracy = []
    server_side_train_loss = []
    server_side_train_accuracy = []

    def fit_config(server_round: int):
        
        config = {
            "server_round": server_round,
            "epochs": 2,
            "proximal_mu": args.proximal_mu,
            "client_kd_alpha": 0.3,
            "client_kd_temperature": 3,
            "server_kd_alpha": 0.7,
            "server_kd_temperature": 7,
            "server_learning_rate": 0.001,}
        
        return config
    

    def eval_config(server_round: int):
        config = {
                "server_round": server_round,
                }
        
        return config
    
    def weightage_average(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]
        return {"accuracy": sum(accuracies) / sum(examples)}
    
    num_classes = 100 if args.name == "cifar100" else 10

    if args.BLIP:
        if args.serverloader_type == "fedmd":
            trainloaders, valloaders, testloader, serverloader = FEDMD_load_dataloaders_blip(args)
        else:
            trainloaders, valloaders, testloader, serverloader = load_dataloaders_blip(args)

        net = MLP4(input_dim=768, hidden_dims=[1024, 512, 256, 128], num_classes=num_classes).to(args.device)

        if args.partitioning == "dirichlet":
            save_path=f"./results/BLIP/{args.algo_type}/{args.alpha}"
        else:
            save_path=f"./results/BLIP/{args.algo_type}/{args.partitioning}"


    else:
        if args.serverloader_type == "fedmd":
            trainloaders, valloaders, testloader, serverloader = FEDMD_load_dataloaders_noblip(args)
        else:
            trainloaders, valloaders, testloader, serverloader = load_dataloaders_noblip(args)

        net = NetS(in_channels=3, num_classes=num_classes).to(args.device)
        
        if args.partitioning == "dirichlet":
            save_path=f"./results/NO_BLIP/{args.algo_type}/{args.alpha}"
        else:
            save_path=f"./results/NO_BLIP/{args.algo_type}/{args.partitioning}"

    
    client_func = gen_client_fn_NOKD(net, trainloaders, valloaders, testloader, args.device, args.algo_type, args.scheduler_)
    

    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar],) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:

        server_train_loss = 0
        server_train_accuracy = 0

        for train_loader in trainloaders:
            train_loss, train_accuracy = test(net, train_loader, args.device)

            server_train_loss += train_loss
            server_train_accuracy += train_accuracy

        server_side_train_loss.append(server_train_loss)
        server_side_train_accuracy.append(server_train_accuracy)

        valloader = testloader
        set_parameters(net, parameters)
        val_loss, val_accuracy = test(net, valloader, args.device)

        print("_"*50)

        print(f"Round --> {server_round} | Server-side evaluation loss: {val_loss} , evaluation accuracy: {val_accuracy}")

        server_side_loss.append(val_loss)
        server_side_accuracy.append(val_accuracy)

        print("_"*50)

        return val_loss, {"accuracy": val_accuracy}
    

    if args.algo_type == "FEDAVG":

        strategy = FedAvg(
            fraction_fit=args.sample_fraction,
            fraction_evaluate=args.sample_fraction,
            min_fit_clients=args.min_num_clients,
            min_available_clients=args.min_num_clients,
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=eval_config,
            evaluate_fn=evaluate,
            evaluate_metrics_aggregation_fn=weightage_average,)
        
        server = Server(strategy=strategy, client_manager=SimpleClientManager())


    elif args.algo_type == "FEDPROX":

        strategy = FedAvg(
            fraction_fit=args.sample_fraction,
            fraction_evaluate=args.sample_fraction,
            min_fit_clients=args.min_num_clients,
            min_available_clients=args.min_num_clients,
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=eval_config,
            evaluate_fn=evaluate,
            evaluate_metrics_aggregation_fn=weightage_average,
            proximal_mu = args.proximal_mu)
        
        server = Server(strategy=strategy, client_manager=SimpleClientManager())


    history = fl.simulation.start_simulation(client_fn=client_func,
                                             server=server,
                                             num_clients = args.num_clients,
                                             config = fl.server.ServerConfig(num_rounds=args.num_rounds),
                                             client_resources={
                                                            "num_cpus": args.num_cpus,
                                                            "num_gpus": args.num_gpus,},
                                            strategy=strategy,)
    

    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, f"history_seed_{args.seed}.pkl"), "wb") as f:
        pickle.dump(history, f)

    print(f"Results saved in {save_path}")

if __name__=="__main__":
    args = parse_arguments()
    run(args)

        
    

        




