import argparse
import torch

def parse_arguments(args=None):
    parser = argparse.ArgumentParser(description="Dataset Partitioning")
    parser.add_argument("--sample_fraction", type=float, default=1.0, help='Fraction of clients to be used for training')
    parser.add_argument("--min_num_clients", type=int, default=2, help='Minimum number of clients to be used for training')
    parser.add_argument("--KD", type=bool, default=False, help="To use knowledge distillation")
    parser.add_argument("--algo_type", type=str, default='FEDMD_homogen', choices=['FEDPROX', 'PROPOSED_heterogen', 'PROPOSED_homogen', 'FEDAVG', 'FEDMD_homogen', 'FEDMD_heterogen', 'scaffold'], help='type of training algorithm')
    #parser.add_argument("--available_architectures", type=lambda s: s.split(','), default=['MLP1','MLP2','MLP3', 'NetS', 'CNN_Hanu', 'CNN_Ram', 'CNN_Lax'], help='Comma-separated model architectures')
    parser.add_argument("--available_architectures_noblip", type=lambda s: s.split(','), default=['NetS','DeepNN_Ram','DeepNN_Hanu', 'DeepNN_Lax'], help='Comma-separated model architectures')
    parser.add_argument("--available_architectures_blip", type=lambda s: s.split(','), default=['MLP1','MLP2','MLP3', 'MLP4'], help='Comma-separated model architectures')
    parser.add_argument("--name", type=str, default='cifar100', choices=['cifar10', 'cifar100', 'tinyimagenet' ], help="Dataset name")
    parser.add_argument("--partitioning", type=str, default='dirichlet', choices=["dirichlet", "label_quantity", "iid", "iid_noniid"], help="Partitioning method")
    parser.add_argument("--num_clients", type=int, default=2, help="Number of clients")
    parser.add_argument("--num_rounds", type=int, default=3, help="Number of rounds")
    parser.add_argument("--BLIP", type=bool, default=False, help="Flag to specify if we are using BLIP")
    parser.add_argument("--proximal_mu", type=float, default=0.02, help="Proximal mu value")
    parser.add_argument("--scheduler_", type=bool, default=False, help="Flag to use scheduler")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation set ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--alpha", type=float, default=1.0, help="Dirichlet alpha value")
    parser.add_argument("--labels_per_client", type=int, default=2, help="Labels per client")
    parser.add_argument("--similarity", type=float, default=0.5, help="Similarity for iid_noniid")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--batch_size_ratio", type=float, default=0.01, help="Batch size ratio")
    parser.add_argument("--server_epochs", type=int, default=5, help="Server epochs")
    parser.add_argument("--num_cpus", type=int, default=1, help="CPUs per client")
    parser.add_argument("--num_gpus", type=int, default=1, help="GPUs per client")                         
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    
    # Safely parse args in Jupyter
    parsed_args, _ = parser.parse_known_args(args)
    return parsed_args


