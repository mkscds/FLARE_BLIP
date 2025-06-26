import os, sys
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

import flwr as fl
from flwr.common import Scalar, parameters_to_ndarray, ndarrays_to_parameters

from Models.models import MLP1, MLP2, MLP3, MLP4, ResNet18
from Models.model_utils import (test,
                                load_models,
                                get_parameters,
                                FEDMD_digest_revisit,
                                get_logits, 
                                save_checkpoints)

class FEDMD_Client(fl.client.NumPyClient):
    def __init__(self, cid:int,
                    trainloader: DataLoader,
                    valloader: DataLoader,
                    testloader: DataLoader,
                    serverloader: DataLoader,
                    device: torch.device,
                    model_architecture:str,
                    algo_type:str,
                    alpha: float,
                    BLIP:bool):
        
        self.cid = cid
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.global_data = serverloader
        self.model_arch = model_architecture
        self.device = device
        self.algo_type = algo_type
        self.alpha = alpha
        self.BLIP = BLIP

        self.local_model, _ = load_models(self.model_arch, self.device, self.algo_type, self.BLIP)

        self.checkpoint_dir = f"{self.algo_type}/{self.alpha}/client_{self.cid}"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.optimizer = torch.optim.Adam(self.local_model.parameters(), lr=1e-3)

    def get_properties(self, config):
        cid = self.cid
        return {'cid': int(cid)}
    
    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return get_parameters(self.local_model)
    
    def fit(self, parameters, config):
        current_round = config.get("server_round", 0)
        #aggregated_logits = parameters_to_ndarray(parameters)
        # load the model and optimizer state from the previous round.
        self._load_checkpoint(current_round - 1)
        # Digest phase
        FEDMD_digest_revisit(self.global_data, self.local_model, self.optimizer, self.device, config, aggregated_logits=parameters, mode="digest")
        # Revisit phase
        FEDMD_digest_revisit(self.trainloader, self.local_model, self.optimizer, self.device, config, aggregated_logits=None, mode="revisit")

        # get logits to communicate to server
        logits = get_logits(self.global_data, self.local_model, self.device)
        # Save checkpoint after training
        self._save_checkpoint(current_round)

        # Validate the model on the validation set
        val_loss, val_accuracy = self._validate(temp=1)

        print(f" Round {current_round} | Client {self.cid} --> Validation Loss: {val_loss:.4f}, Validation Accuracy: {100*val_accuracy:.2f}%")

        return [logits], len(self.trainloader.dataset), {}
    
    def _validate(self, temp=1):
        """Validate the model on the validation set."""
        return test(self.local_model, self.valloader, self.device)
    
    def evaluate(self, parameters, config):
        current_round = config.get("server_round", 0)
        self._load_checkpoint(current_round)
        loss, acc = test(self.local_model, self.testloader, self.device)
        return float(loss), {"accuracy": float(acc)}
    

    
    def _save_checkpoint(self, round_num: int):
        save_checkpoints(self.local_model, self.optimizer, self.checkpoint_dir, round_num)


    def _load_checkpoint(self, round_num: int):
        checkpoint = torch.load(
            os.path.join(self.checkpoint_dir, f"checkpoint_round_{round_num-1}.pt"),
            map_location="cpu")
        
        self.local_model.load_state_dict(checkpoint["model"]).to(self.device)
        self.optimizer.load_state_dict(checkpoint["optimizer"])



def FEDMD_client_fn(trainloaders: List[DataLoader],
                    valloaders: List[DataLoader],
                    testloader: DataLoader,
                    server_loader: DataLoader,
                    device: torch.device,
                    model_architectures: List[int],
                    algo_type: str,
                    alpha: float,
                    BLIP: bool,
                    ):
    def client_fn(cid:str) -> FEDMD_Client:
    # Create and return the client
        return FEDMD_Client(cid=int(cid),
                        trainloader=trainloaders[cid],
                        valloader=valloaders[cid],
                        testloader=testloader,
                        serverloader=server_loader,
                        device=device,
                        model_architecture=model_architectures[cid],
                        algo_type=algo_type,
                        alpha=alpha,
                        BLIP=BLIP)
    
    return client_fn


        


    