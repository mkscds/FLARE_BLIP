from Common.dataset_fedmd import FEDMD_load_datasets_noblip
from Config import parse_arguments


if __name__=="__main__":
    args = parse_arguments()
    train_datasets, val_datasets, testset, serverset = FEDMD_load_datasets_noblip(args)