from typing import Callable, Dict,  List, OrderedDict
import os, sys
import flwr as fl
from flwr.common import Scalar, Context

import torch
from torch.utils.data import DataLoader
from Models.models import MLP1, MLP2, MLP3, MLP4, ResNet18
from Models.model_utils import train, train_KD, test, load_models, set_parameters, get_parameters


class ClientProposed(fl.client.NumPyClient):
    def __init__(self, cid: int, trainloader: DataLoader, valloader: DataLoader, testloader: DataLoader, device: torch.device, model_architecture: str, algo_type: str, BLIP: bool) -> None:
        super().__init__()

        self.cid = cid
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.model_architecture = model_architecture
        self.device = device
        self.algo_type = algo_type
        self.BLIP = BLIP

        self.client_model, self.server_model = load_models(self.model_architecture, self.device, self.algo_type, self.BLIP)

        