import os
import tqdm as tqdm

from logging import WARNING
from typing import Callable, Optional, List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from flwr.common import (FitRes, MetricsAggregationFn, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from Models.model_utils import (get_parameters, set_parameters, load_models, select_model)

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """ """

class CustomProposed(FedAvg):
    def __init__(
        self,
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
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        server_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
        client_architectures: Optional[List[str]] = None,
        algo_type: Optional[str] = None,
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )

        self.accept_failures = accept_failures
        self.server_loader = server_loader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.client_architectures = client_architectures or []
        self.algo_type = algo_type

        if not self.client_architectures:
            raise ValueError("Client architectures list cannot be empty")
        
        

