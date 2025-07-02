from typing import List
from collections import OrderedDict
import os, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import flwr as fl
from flwr.common import ndarrays_to_parameters
from .models import MLP1, MLP2, MLP3, MLP4, ResNet18, NetS, DeepNN_Ram, DeepNN_Hanu, DeepNN_Lax
from torch.utils.data import DataLoader
# from .models import CNN_3layer_fc_model_removelogsoftmax, CNN_2layer_fc_model_removelogsoftmax, ResNet18, Net



def get_param_size(params):
    return sum(p.nbytes for p in fl.common.parameters_to_ndarrays(params)) / (1024 * 1024)

def fedprox_loss_term(proximal_mu, net, global_params):
    proximal_term = 0.0

    if proximal_mu == 0.0:
        return proximal_term
    else:
        for local_weights, global_weights in zip(net.parameters(), global_params):
            proximal_term += (local_weights - global_weights).norm(2)
        proximal_term = (proximal_mu / 2) * proximal_term 
        return proximal_term
    

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def proximal_loss_func(proximal_mu, net, global_params):
    proximal_term = 0.0

    if proximal_mu == 0.0:
        return proximal_term
    else:
        for local_weights, global_weights in zip(net.parameters(), global_params):
            proximal_term += (local_weights - global_weights).norm(2)
        proximal_term = (proximal_mu / 2) * proximal_term 
        return proximal_term

def save_checkpoints(net, optimizer, current_round, checkpoint_dir):
        torch.save({
            "model": net.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        os.path.join(checkpoint_dir, f"checkpoint_round_{current_round}.pt"))

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict()
    for k, v in params_dict:
        # Handle scalar tensors (like num_batches_tracked) properly
        if v.ndim == 0:  # Scalar array
            state_dict[k] = torch.tensor(v)
        else:
            state_dict[k] = torch.tensor(v)
    net.load_state_dict(state_dict, strict=True)

class ScaffoldOptimizer(torch.optim.SGD):
    def __init__(self, grads, step_size, momentum, weight_decay):
        super().__init__(grads, lr=step_size, momentum=momentum, weight_decay=weight_decay)
    def step_custom(self, server_cv, client_cv):
        self.step()
        for group in self.param_groups:
            for par, s_cv, c_cv in zip(group['params'], server_cv, client_cv):
                par.data.add_(s_cv - c_cv, alpha=-group['lr'])
                
def train_scaffold(net: nn.Module, trainloader: DataLoader, learning_rate: float, momentum: float, weight_decay: float,  epochs: int, device: torch.cuda.device, server_cv: list, client_cv: list):
    """Train the network on the training set using SCAFFOLD."""
    criterion = nn.CrossEntropyLoss()
    optimizer = ScaffoldOptimizer(net.parameters(), learning_rate, momentum, weight_decay)
    net.train()
    for _ in range(epochs):
        for data, target in trainloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step_custom(server_cv, client_cv)



def train(net, trainloader, optimizer, device, config, cid): 
    """Train the network on the training set."""
    epochs = config.get("epochs", 1)
    server_round = config.get("server_round", 0)
    scheduler_ = config.get("scheduler_", False)
    proximal_mu = config.get("proximal_mu", 0.0)

    add_epoch_acc = 0
    add_epoch_loss = 0
    criterion = torch.nn.CrossEntropyLoss()
    #optimizer = torch.optim.SGD(net.parameters(),lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    global_params = [param.detach().clone() for param in net.parameters()]

    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels) + proximal_loss_func(proximal_mu, net, global_params)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        before_lr = optimizer.param_groups[0]["lr"]
        if scheduler_ and epoch!=epochs-1:
            scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]
        
        add_epoch_loss+=epoch_loss
        add_epoch_acc+=epoch_acc
            
        print(f"Round --> {server_round} | Client --> {cid} | Epoch {epoch+1}: train loss {epoch_loss}, train accuracy {epoch_acc}")


def FEDMD_train(net, trainloader, epochs: int, device: torch.cuda.device, server_rounds:int, scheduler_:bool, proximal_mu: float, mode: str):
    """Train the network on the training set."""
    add_epoch_acc = 0
    add_epoch_loss = 0
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(),lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    global_params = [param.detach().clone() for param in net.parameters()]

    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(net(images), labels) + proximal_loss_func(proximal_mu, net, global_params)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        before_lr = optimizer.param_groups[0]["lr"]
        if scheduler_ and epoch!=epochs-1:
            scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]
        
        add_epoch_loss+=epoch_loss
        add_epoch_acc+=epoch_acc
            
        print("Round %d :Epoch %d: SGD lr %f -> %f" % (server_rounds,epoch, before_lr, after_lr))
        print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")


    return net, optimizer


def get_logits(dataloader, net, device):
    logits = []
    net.eval()
    with torch.inference_mode():
        for images, _ in dataloader:
            images = images.to(device)
            outputs = net(images)
            logits.append(outputs.cpu().numpy())

    logits = np.concatenate(logits, axis=0)
    return logits

def FEDMD_digest_revisit(dataloader, net, optimizer, device, config, aggregated_logits=None, mode="revisit"):
    epochs = config.get("epochs", 1)
    server_rounds = config.get("server_round", 0)
    proximal_mu = config.get("proximal_mu", 0.0)
    alpha = config.get("client_KD_alpha", 0.1)
    temperature = config.get("client_KD_temperature", 1.0)
    global_params = [params.detach().clone() for params in net.parameters()]
    Epoch_Loss = []
    Epoch_Accuracy = []

    def kd_loss(student_logits, labels, teacher_logits):
        loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(student_logits / temperature, dim=1), F.softmax(teacher_logits / temperature, dim=1)) * (temperature ** 2) + \
                                                F.cross_entropy(student_logits, labels) * (1 - alpha)
        return loss
    
    net.train()    
    
    if mode == "digest" and aggregated_logits is not None:
        aggregated_logits = aggregated_logits.to(device)
        for epoch in range(epochs):
            for batch_idx, (images, labels) in enumerate(dataloader):
                teacher_logits = None
                images, labels = images.to(device), labels.to(device)
                batch_size = labels.size(0)
                optimizer.zero_grad()
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                student_logits = net(images)
                teacher_logits = aggregated_logits[start_idx: end_idx]
                loss = kd_loss(student_logits, labels, teacher_logits)
                loss.backward()
                optimizer.step()

    elif mode == "revisit":
        criterion = nn.CrossEntropyLoss()
        scheduler_ = config.get("scheduler", None)
        if scheduler_:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        for epoch in range(epochs):
            correct, total, epoch_loss = 0, 0, 0.0
            for batch_idx, (images, labels) in enumerate(dataloader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels) + proximal_loss_func(proximal_mu, net, global_params)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                total += labels.size(0)
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            epoch_loss /= len(dataloader.dataset)
            epoch_acc = 100. * correct / total
            if scheduler_ and epoch!=epochs-1:
                scheduler.step()
            
            Epoch_Loss.append(epoch_loss)
            Epoch_Accuracy.append(epoch_acc)

            print(f" Round {server_rounds} --> Epoch {epoch + 1}/{epochs} \t Loss: {epoch_loss:.4f} \t Accuracy: {epoch_acc:.2f}%")


def average_client_logits(stacked_array: np.ndarray, weighted: bool = False, weights: np.ndarray = None, num_clients: int = 10, num_samples: int = 5000):
    """
    Averages logits from multiple clients.

    Args:
        stacked_array (np.ndarray): Stacked logits of shape (num_clients * num_samples, num_classes).
        weighted (bool): Whether to perform weighted averaging. Default is False.
        weights (np.ndarray): Array of shape (num_clients,) specifying client weights. Required if weighted=True.

    Returns:
        np.ndarray: Averaged logits of shape (num_samples, num_classes).
    """

    if len(stacked_array.shape) == 3:
        reshaped = stacked_array

    else:
        # Reshape to (num_clients, 5000, num_classes)
        num_classes = stacked_array.shape[1]
        reshaped = stacked_array.reshape(num_clients, num_samples, num_classes)

    if weighted:
        if weights is None or len(weights) != num_clients:
            raise ValueError("Weights must be provided and match number of clients.")
        weights = np.array(weights).reshape(num_clients, 1, 1)
        averaged = np.sum(reshaped * weights, axis=0) / np.sum(weights)
    else:
        averaged = reshaped.mean(axis=0)
    
    return averaged


# def test(net, testloader, device, temp=1.0):
#     """Evaluate the network on the entire test set."""
#     criterion = torch.nn.CrossEntropyLoss()
#     correct, total, loss = 0, 0, 0.0
#     net.eval()
#     with torch.no_grad():
#         for images, labels in testloader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = net(images)
#             loss += criterion(outputs, labels).item()
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     loss /= len(testloader.dataset)
#     accuracy = correct / total
#     return loss, accuracy

def test(net, testloader, device, temp=1.0):
    """
    Evaluate the network on the test set batch-wise.
    
    Args:
        net: the neural network
        testloader: DataLoader for test data
        device: torch.device
        temp: temperature scaling factor (default 1.0)
    
    Returns:
        Tuple: (average loss, accuracy)
    """
    criterion = torch.nn.CrossEntropyLoss()
    net.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images) / temp  # if temp != 1.0, apply temperature scaling
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size  # accumulate batch loss
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return average_loss, accuracy


    
###############################################################################################
# Utils for KD based algorithms
###############################################################################################

def create_model_architectures_noblip(available_architectures_noblip, algo_type, num_clients):
    if algo_type in  ["PROPOSED_homogen", "FEDMD_homogen", "FEDAVG"]:
        selected_model = "NetS"
        model_architectures = {i: selected_model for i in range(num_clients)}

    elif algo_type in ["FEDMD_heterogen", "PROPOSED_heterogen"]:
        model_architectures = {i: random.choices(available_architectures_noblip, weights=[0.3, 0.6, 0.1])[0] for i in range(num_clients)}

    else:
        raise ValueError(f"Unknown algorithm type: {algo_type}")
    
    return model_architectures


def create_model_architectures_blip(available_architectures_blip, algo_type, num_clients):
    if algo_type in  ["homogeneous", "FEDMD_homogen"]:
        selected_model = "MLP4"
        model_architectures = {i: selected_model for i in range(num_clients)}

    elif algo_type in ["FEDMD_heterogen", "heterogeneous"]:
        model_architectures = {i: random.choices(available_architectures_blip, weights=[0.3, 0.6, 0.1])[0] for i in range(num_clients)}

    else:
        raise ValueError(f"Unknown algorithm type: {algo_type}")
    
    return model_architectures



    


def select_model(model_architecture: str, device: torch.device, BLIP:bool, num_classes:int):
    if BLIP:
        if model_architecture == "MLP1":
            net = MLP1().to(device)
        elif model_architecture == "MLP2":
            net = MLP2().to(device)
        elif model_architecture == "MLP3":
            net = MLP3().to(device)
        elif model_architecture == "MLP4":
            net = MLP4().to(device)
        else:
            raise ValueError(f"Unknown model architecture: {model_architecture}")
    else:
        if model_architecture == "NetS":
            net = NetS(in_channels=3, num_classes=num_classes).to(device)
        elif model_architecture == "DeepNN_Ram":
            net = DeepNN_Ram(in_channels=3, num_classes=num_classes).to(device)
        elif model_architecture == "DeepNN_Hanu":
            net = DeepNN_Ram(in_channels=3, num_classes=num_classes).to(device)
        elif model_architecture == "DeepNN_Lax":
            net = DeepNN_Ram(in_channels=3, num_classes=num_classes).to(device)
        else:
            raise ValueError(f"Unknown model architecture: {model_architecture}")

    
    return net

def load_models(model_architecture: str, device: torch.device, algo_type: str, BLIP: bool, num_classes:int):

    client_model = select_model(model_architecture, device, BLIP, num_classes)

    if algo_type == "PROPOSED_homogen":
        server_model = client_model.__class__().to(device)

    elif algo_type == "PROPOSED_heterogen":
        if BLIP:
            server_model = MLP4(num_classes=num_classes).to(device)
        else:
            server_model = ResNet18(num_classes).to(device)

    elif algo_type in ["fedavg", "fedbn", "fedprox"]:
        server_model = None

    elif algo_type in ["FEDMD_homogen", "FEDMD_heterogen"]:
        server_model = None
    else:
        raise ValueError(f"Unknown algo_type: {algo_type}")
    
    return client_model, server_model


def train_KD(student_model, teacher_model, config, trainloader, optimizer, device):
    epochs = config["epochs"]
    alpha = config["client_kd_alpha"]
    temperature = config["client_kd_temperature"]

    def kd_loss(student_output, labels, teacher_output):
        # KD loss
        return nn.KLDivLoss(reduction='batchmean')(F.log_softmax(student_output/temperature, dim=1), F.softmax(teacher_output/temperature, dim=1)) * (alpha * temperature**2) + \
               F.cross_entropy(student_output, labels) * (1 - alpha)
    
    
    student_model.train()
    teacher_model.eval()

    for epoch in range(epochs):  # Track epoch number
        running_loss = 0.0
        running_correct = 0
        total_samples = 0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward pass through student and teacher
            s_out = student_model(images)
            with torch.inference_mode():
                t_out = teacher_model(images)

            # Compute loss and backpropagate
            loss = kd_loss(s_out, labels, t_out)
            loss.backward()
            optimizer.step()

            # Accumulate metrics
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            _, preds = torch.max(s_out, 1)  # Get predictions
            running_correct += (preds == labels).sum().item()
            total_samples += batch_size

        # Calculate epoch metrics
        epoch_loss = running_loss / total_samples
        epoch_acc = running_correct / total_samples

        print(f'Epoch [{epoch+1}/{epochs}]: '
              f'Train Loss: {epoch_loss:.4f}, '
              f'Train Accuracy: {epoch_acc*100:.2f}%')
        


def get_server_initial_parameters(net, server_dataloader, server_epochs, device: str):
    optimizer = torch.optim.SGD(net.parameters(),lr=0.001)
    print('\n INITIALIZING SERVER MODEL \n')
    train(net.to(device), server_dataloader, server_epochs, device, server_rounds=0, scheduler_=False, proximal_mu=0.0)
    print('\n COMPLETED INITIALIZING SERVER MODEL \n')
    numpy_parameters = get_parameters(net)
    # server_initial_parameters = fl.common.ndarrays_to_parameters(numpy_parameters)

    return numpy_parameters


