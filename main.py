import sys
sys.modules['tkinter'] = None
import os
import pickle
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from torch.utils.data import Dataset, DataLoader

import flwr as fl
from typing import Callable, Dict, List, OrderedDict, Tuple, Optional
from Config import parse_arguments
from Common.dataset_fedmd import FEDMD_load_dataloaders_blip, FEDMD_load_datasets_noblip, FEDMD_load_dataloaders_noblip
from Models.models import BLIPVisionWrapper, NetS, MLP1, MLP2, MLP4
from client import gen_client_fn_NOKD
from Models.model_utils import test, set_parameters


def run(args):

    server_side_loss = []
    server_side_accuracy = []
    server_side_train_loss = []
    server_side_train_accuracy = []

    def fit_config(server_round: int):
        """Return training configuration dict for each round.

        Perform two rounds of training with one local epoch, increase to two local
        epochs afterwards.
        """
        config = {
            "server_round": server_round,  # The current round of federated learning
            "epochs": 10,
            'proximal_mu':0.0,
            "client_kd_alpha": 0.3,
            "client_kd_temperature": 3,
            "server_kd_alpha": 0.7,
            "server_kd_temperature": 7,
            "server_learning_rate": 0.0001}
        return config

    def eval_config(server_round: int):
        """
        Generate configuration parameters for the client's evaluate operation.
        """
        config = {
            "server_round": server_round, # the current round of federated learning

        }
        return config



    def weighted_average(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
        """Aggregate client accuracies by number of test samples"""
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]
        return {"accuracy": sum(accuracies) / sum(examples)}
    
    # The `evaluate` function will be by Flower called after every round


    if args.BLIP:
        trainloaders, valloaders, testloader, server_loader = FEDMD_load_dataloaders_blip(args)
        net = MLP4(input_dim=768, hidden_dims=[1024, 512, 256, 128], num_classes=100).to(args.device)
        save_path = f"./results/BLIP/{args.algo_type}/{args.alpha}"
    else:
        trainloaders, valloaders, testloader, server_loader = FEDMD_load_dataloaders_noblip(args)
        # Load model
        net = NetS(in_channels=3, num_classes=100).to(args.device)
        save_path = f"./results/NO_BLIP/{args.algo_type}/{args.alpha}"   

    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar],) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        
        server_train_loss = 0
        server_train_accuracy = 0
        for train_loader in trainloaders:
            train_loss, train_accuracy = test(net, train_loader, args.device, temp=1.0) 
    #         print(f"Server-side Train loss {train_loss} / accuracy {train_accuracy}")
            server_train_loss+=train_loss
            server_train_accuracy+=train_accuracy
        # print("_"*50)
        server_side_train_loss.append(server_train_loss/len(trainloaders))
        server_side_train_accuracy.append(server_train_accuracy/len(trainloaders))
        # print(server_side_train_loss,server_side_train_accuracy)
        # print("_"*50)
        
        valloader = testloader
        set_parameters(net, parameters)  # Update model with the latest parameters
        val_loss, val_accuracy = test(net, valloader, args.device)
        print(f"Server-side evaluation loss {val_loss} / accuracy {val_accuracy}")
        server_side_loss.append(val_loss)
        server_side_accuracy.append(val_accuracy)
        print("_"*50)
        print(server_side_loss)
        print(server_side_accuracy)
        print("_"*50)
            
        return val_loss, {"accuracy": val_accuracy}

    client_func = gen_client_fn_NOKD(net, trainloaders, valloaders, testloader, args.device)

    strategy = fl.server.strategy.FedAvg(
                            fraction_fit=1.0,
                            fraction_evaluate=1.0,
                            min_fit_clients=args.num_clients,
                            min_evaluate_clients=args.num_clients,
                            min_available_clients=args.num_clients,
                            evaluate_fn=evaluate,
                            on_fit_config_fn=fit_config,
                            evaluate_metrics_aggregation_fn=weighted_average,)
    

    history = fl.simulation.start_simulation(
                        client_fn=client_func,
                        num_clients = args.num_clients,
                        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
                        client_resources={
                            "num_cpus": args.num_cpus,
                            "num_gpus": args.num_gpus,},
                        strategy=strategy,)
    


    # save_path = f"./results/NO_BLIP/{args.algo_type}/{args.alpha}"
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, f"history_{args.alpha}.pkl"), "wb") as f_ptr:
        pickle.dump(history, f_ptr)

    print(f"Results saved in {save_path}")



if __name__=='__main__':
    args = parse_arguments()
    run(args)