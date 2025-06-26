from typing import Callable, Dict, List, OrderedDict
import os, sys
import flwr as fl
import torch
from flwr.common import Scalar
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Models.model_utils import train, test, get_parameters, set_parameters
from torch.utils.data import DataLoader


class FlowerClientNOKD(fl.client.NumPyClient):
    def __init__(self, net, cid, trainloader, valloader, testloader, device, scheduler_):
        self.cid = cid
        # self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.device = device
        self.net = net
        self.scheduler_ = scheduler_
        self.optimizer = torch.optim.SGD(
            self.net.parameters(),
            lr=0.0001,
            momentum=0.9,
            weight_decay=0.1
        )



    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return get_parameters(self.net)

    def fit(self, parameters, config):
        server_round = config["server_round"]
        local_epochs = config["epochs"]
        proximal_mu = config["proximal_mu"]

        print(f"[Client {self.cid}, round {server_round}] fit, config: {config}")
        set_parameters(self.net, parameters)
    
        train(self.net, self.trainloader, local_epochs, self.device, server_round, scheduler_=self.scheduler_, proximal_mu=proximal_mu)
        # train(self.net, self.trainloader,local_epochs,server_round,)
        val_loss, val_acc = self._validate()
        print(f"Client {self.cid} validation - Loss: {val_loss:.4f}, Acc: {(val_acc*100):.2f}%")
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        set_parameters(self.net, parameters)       
        loss, accuracy = test(self.net, self.testloader, self.device, temp=1.0)
        return float(loss), len(self.testloader), {"accuracy": float(accuracy)}
    
    def _validate(self):
        """Run validation on local validation set"""
        return test(self.net, self.valloader, self.device, temp=1.0)    

def gen_client_fn_NOKD(
    net,
    trainloaders: list[DataLoader],
    valloaders: List[DataLoader],
    testloader: DataLoader,  # Global test set
    device: torch.device,
    scheduler_: bool = False,
    ):
    def client_fn(cid: str) -> FlowerClientNOKD:
        return FlowerClientNOKD(
            net=net,
            cid=int(cid),
            trainloader=trainloaders[int(cid)],
            valloader=valloaders[int(cid)],
            testloader=testloader,  # Same for all clients
            device=device,
            scheduler_=scheduler_
        ).to_client()
    return client_fn

