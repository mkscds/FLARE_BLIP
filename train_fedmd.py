import sys
sys.modules['tkinter'] = None
import os, copy
import pickle
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset

import flwr as fl
from flwr.server.client_manager import SimpleClientManager
from flwr.server.server import Server


from typing import Callable, Dict, List, OrderedDict, Tuple, Optional
from Config import parse_arguments
from Common.dataset_fedmd import FEDMD_load_dataloaders_blip, FEDMD_load_dataloaders_noblip
from Models.model_utils import create_model_architectures_noblip, create_model_architectures_blip, select_model, FEDMD_digest_revisit, save_checkpoints, get_logits, average_client_logits
from Models.models import MLP1, MLP2, MLP3, MLP4, ResNet18

from strategy_fedmd import FEDMD_Strategy
from client_fedmd import FEDMD_client_fn


    
def run(args):

    server_side_loss = []
    server_side_accuracy = []
    server_side_train_loss = []
    server_side_train_accuracy = []

    num_classes = 100 if args.name == "cifar100" else 10

    def fit_config(server_round: int):
        """Return training configuration dict for each round."""

        config = {
            "server_round": server_round,
            "epochs": 2,
            "proximal_mu":args.proximal_mu,
            "client_kd_alpha": 0.3,
            "client_kd_temperature": 3,
            "server_kd_alpha": 0.7,
            "server_kd_temperature": 7,
            "server_lr": 0.001 
             }
        return config
    

    def eval_config(server_round: int):
        """
        Generate configuration parameters for the client's evaluate operation.
        """

        config = {"server_round": server_round}

        return config
    

    def weighted_average(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
        """Aggregate client accuracies by number of test samples"""
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]
        return {"accuracy": sum(accuracies) / sum(examples)}


    def evaluate(server_round, parameters, config):

        server_train_loss = 0.0
        server_train_accuracy = 0.0

        print("\t No server model for fedmd")

        return 0.0, {"accuracy": 0.1}
    
    def trasfer_learning_init(args, model_architectures, serverloader, trainloaders):
        """
        Initialize the model for transfer learning.
        This function trains the model on the server data and then initializes client models.
        """

        client_logits = {}
        config = fit_config(server_round=0)

        for i, trainloader in enumerate(trainloaders):
            print(f"\n Transfer learning on client {i + 1}\n")

            net = select_model(model_architectures[i], args.device, args.BLIP, num_classes=num_classes)
            optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
            # Training the model on public data
            FEDMD_digest_revisit(serverloader, net, optimizer, args.device, config, aggregated_logits=None, mode="revisit")
            # Training the model on private data
            FEDMD_digest_revisit(trainloader, net, optimizer, args.device, config, aggregated_logits=None, mode="revisit")
            # logit Communication for round 1
            client_logits[i] = get_logits(serverloader, net, args.device)
            # Save the model and optimizer state for each client
            checkpoint_dir = f"client_checkpoints/{args.algo_type}/{args.alpha}/client_{i}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            save_checkpoints(net, optimizer, 0, checkpoint_dir)

        logits = np.vstack([client_logits[i] for i in sorted(client_logits.keys())])
        aggregated_logits = average_client_logits(logits, num_clients=args.num_clients, num_samples=5000)

        return aggregated_logits

    if args.algo_type in ["FEDMD_homogen", "FEDMD_heterogen"]:
        if args.BLIP:
            trainloaders, valloaders, testloader, server_loader = FEDMD_load_dataloaders_blip(args)
            results_save_path = f"./results/BLIP/{args.algo_type}/{args.alpha}"
            model_architectures = create_model_architectures_blip(args.available_architectures_blip, args.algo_type, args.num_clients, args.BLIP)
        else:
            trainloaders, valloaders, testloader, server_loader = FEDMD_load_dataloaders_noblip(args)
            results_save_path = f"./results/NO_BLIP/{args.algo_type}/{args.alpha}"
            model_architectures = create_model_architectures_noblip(args.available_architectures_noblip, args.algo_type, args.num_clients)
        # get averaged logits for round 1
        logits = trasfer_learning_init(args, model_architectures, server_loader, trainloaders)

        client_func = FEDMD_client_fn(trainloaders, valloaders, testloader, server_loader, args.device, model_architectures, args.algo_type, args.alpha, args.BLIP, num_classes)

        

        strategy = FEDMD_Strategy(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=args.num_clients,
            min_evaluate_clients=args.num_clients,
            min_available_clients=args.num_clients,
            initial_parameters=fl.common.ndarrays_to_parameters(logits),
            evaluate_fn=evaluate,
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=eval_config,
            evaluate_metrics_aggregation_fn=weighted_average,
            client_architectures=model_architectures,
            algo_type=args.algo_type,
            server_loader=server_loader,
            num_clients=args.num_clients,
        
        )

    server = Server(strategy=strategy, client_manager=SimpleClientManager())
    
    history = fl.simulation.start_simulation(
                        client_fn=client_func,
                        server=server,
                        num_clients = args.num_clients,
                        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
                        client_resources={
                            "num_cpus": args.num_cpus,
                            "num_gpus": args.num_gpus,},
                        strategy=strategy,)
    

    os.makedirs(results_save_path, exist_ok=True)
    with open(os.path.join(results_save_path, f"history_{args.seed}.pkl"), "wb") as f_hist:
        pickle.dump(history, f_hist)

    print(f"Results saved in {results_save_path}")


if __name__=="__main__":
    args = parse_arguments()
    run(args)






        

    


        











    


    
