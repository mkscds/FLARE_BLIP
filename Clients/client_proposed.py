import os, sys
from typing import Callable, Dict, List, OrderedDict

import torch
from torch.utils.data import DataLoader

import flwr as fl
from flwr.common import Scalar

from Models.models import MLP1, MLP2, MLP3, MLP4, ResNet18
from Models.model_utils import test, load_models, set_parameters, get_parameters, train_KD

class ClientProposed(fl.client.NumPyClient):
    def __init__(self,
                 cid,
                 trainloader,
                 valloader,
                 testloader,
                 device,
                 model_architecture,
                 algo_type,
                 BLIP,
                 num_classes) -> None:
        
        self.cid = cid
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.device = device
        self.model_architecture = model_architecture
        self.algo_type = algo_type
        self.BLIP = BLIP
        self.num_classes = num_classes

        self.client_model, self.global_model = load_models(self.model_architecture, self.device, self.algo_type, self.num_classes)

        self.checkpoint_dir = f"client_checkpoint/{self.algo_type}/client_{self.cid}"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.optimizer = torch.optim.Adam(self.client_model.parameters(), lr=1e-3)

    def get_properties(self, config) -> Dict[str, Scalar]:
        cid = self.cid
        return {'cid': int(cid)}
    
    def get_parameters(self, config):
        return get_parameters(self.client_model)
    
    def fit(self, parameters, config):
        server_round = config["server_round"]
        local_epochs = config["epochs"]
        if self.algo_type == "fedprox":
            proximal_mu = config["proximal_mu"]
        else:
            proximal_mu = 0.0

        print(f"Round --> {server_round} | Client --> {self.cid} | Fit Config --> {config}")       

        set_parameters(self.global_model, parameters)

        if server_round == 1:
            train_KD(self.client_model, self.global_model, config, self.trainloader, self.optimizer, self.device)
            self._save_checkpoint(server_round)

        else:
            self._load_checkpoint(server_round - 1)
            train_KD(self.client_model, self.global_model, config, self.trainloader, self.optimizer, self.device)
            self._save_checkpoint(server_round)
        
        val_loss, val_accuracy = self._validate()
        
        print(f" Round {server_round} | Client {self.cid} --> Validation Loss: {val_loss:.4f}, Validation Accuracy: {100*val_accuracy:.2f}%")

        return self.get_parameters(config), len(self.trainloader.dataset), {"val_acc": val_accuracy, "cid": float(self.cid)}
    

    def evaluate(self, parameters, config):
        current_round = config["server_round"]
        self._load_checkpoint(current_round)
        loss, acc = test(self.client_model, )


