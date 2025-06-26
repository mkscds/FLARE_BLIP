import os
from tqdm import tqdm
from logging import WARNING
from typing import Callable, Optional, Union, List, Tuple, Dict

from flwr.common import (EvaluateIns, EvaluateRes, FitIns, FitRes, MetricsAggregationFn, NDArrays, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays)

from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from flwr.server.strategy.aggregate import aggregate, aggregate_inplace, weight_loss_avg
from flwr.server.strategy import FedAvg

from Models.model_utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """ """

class FEDMD_Strategy(FedAvg):
    def __init__(self,
                 fraction_fit: float = 1,
                 fraction_evaluate: float = 1,
                 min_fit_clients: int = 2,
                 min_evaluate_clients: int = 2,
                 min_available_clients: int = 2,
                 evaluate_fn: Optional[Callable] = None,
                 on_fit_config_fn: Optional[Callable] = None,
                 on_evaluate_config_fn: Optional[Callable] = None,
                 accept_failures: bool = True,
                 initial_parameters: Optional[Parameters] = None,
                 fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
                 evaluate_metrics_aggregate_fn: Optional[MetricsAggregationFn] = None,
                 server_loader: Optional[DataLoader] = None,
                 device: Optional[torch.device] = None,
                 client_architectures: Optional[List[str]] = None,
                 algo_type: Optional[str] = None):
        

        super().__init__(
            fraction_fit = fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregate_fn=evaluate_metrics_aggregate_fn
        )

        self.initial_parameters = initial_parameters
        self.accept_failures = accept_failures
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.client_architectures = client_architectures
        self.algo_type=algo_type
        self.serverloader = server_loader
        self.num_samples = len(self.serverloader.dataset)

        if not self.client_architectures:
            raise ValueError("Client archiectures list cannot be empty")
        

        def __repr__(self) -> str:
            return f"FEDMD(accept_failures={self.accept_failures})"
       

        def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
            """Initialize global model parameters"""

            np_parameters = parameters_to_ndarrays(self.initial_parameters)
            
            initial_parameters = average_client_logits(np_parameters, num_clients = 10, num_samples=self.num_samples)
            self.initial_parameters = None

            return initial_parameters
        

        def evaluate(self, server_round: int, parameters: Parameters) -> Optional[tuple[float, dict[str, Scalar]]]:
            """Evaluate model parameters using an evaluation function"""
            if self.evaluate_fn is None:
                return None
            parameters_ndarrays = parameters_to_ndarrays(parameters)
            eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
            if eval_res is None:
                return None
            loss, metrics = eval_res

            return loss, metrics
        

        def aggregate_fit(self, server_round: int, results: list[tuple[ClientProxy, FitRes]], failures: list[Union[tuple[ClientProxy, FitRes], BaseException]]):
            """Aggregate fit results using weighted average."""

            if not results:
                return None, {}
            
            if not self.accept_failures and failures:
                return None, {}
            
            param_results = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]

            np_param_results = np.array(param_results)

            aggregated_ndarrays = average_client_logits(np_param_results, num_clients = 10, num_samples=self.num_samples)


            parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

            metrics_aggregated = {}

            if self.fit_metrics_aggregation_fn:
                fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
                metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
            elif server_round == 1:
                log(WARNING, "No fit_metrics_aggregation_fn provided")


            return parameters_aggregated, metrics_aggregated
        


        





