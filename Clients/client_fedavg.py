from typing import List

import os, sys
import flwr as fl
from flwr.common import Scalar


import torch
from torch.utils.data import DataLoader

from Models.model_utils import train, test, get_parameters, set_parameters

class FlowerClientNOKD(fl.client.NumPyClient):
    def __init__(self, net, cid, trainloader, valloader, testloader, device, algo_type, scheduler_):
        self.cid = cid
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.device = device
        self.net = net
        self.scheduler_ = scheduler_
        self.algo_type = algo_type
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)

    def get_parameters(self, config):
        return get_parameters(self.net)
    

    def fit(self, parameters, config):
        server_round = config["server_round"]
        local_epochs = config["epochs"]
        if self.algo_type == "fedprox":
            proximal_mu = config["proximal_mu"]
        else:
            proximal_mu = 0.0

        print(f"Round --> {server_round} | Client --> {self.cid} | Fit Config --> {config}")

        set_parameters(self.net, parameters)
        
        train(self.net, self.trainloader, self.optimizer, self.device, config, self.cid)

        val_loss, val_acc = self._validate()
        
        print(f"Round --> {server_round} | Client --> {self.cid} | Validation Loss: {val_loss:.4f} , Validation Accuracy: {(val_acc*100):.2f}")

        return get_parameters(self.net), len(self.trainloader.dataset), {}
    

    def evaluate(self, parameters, config):
        server_round = config["server_round"]
        print(f"Round --> {server_round} | Client --> {self.cid} | Evaluate Config --> {config}")
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.testloader, self.device, temp=1.0)

        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}
    

    def _validate(self):
        return test(self.net, self.valloader, self.device, temp=1.0)
    
def gen_client_fn_NOKD(net, trainloaders: list[DataLoader], valloaders: list[DataLoader], testloader: DataLoader, device: torch.device, algo_type: str, scheduler_: bool=False):
    def client_fn(cid: str) -> FlowerClientNOKD:
        return FlowerClientNOKD(net=net,
                                cid=int(cid),
                                trainloader=trainloaders[int(cid)],
                                valloader=valloaders[int(cid)],
                                testloader=testloader,
                                device=device,
                                algo_type = algo_type,
                                scheduler_=scheduler_).to_client()
    return client_fn

