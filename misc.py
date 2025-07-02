import sys
import pickle

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from data_utils import load_CIFAR_data, load_CIFAR_from_local, generate_partial_data, generate_imbal_CIFAR_private_data



def paseArg():
    pass

if __name__=="__main__":
    conf_file = parseArg()
    with open(conf_file, 'r') as f:
        conf_dict = eval(f.read())


        model_config = conf_dict["models"]
        pre_train_params = conf_dict["pre_train_params"]
        model_saved_dir = conf_dict["model_saved_dir"]

        public_classes = conf_dict["public_classes"]
        public_classes.sort()
        private_classes = conf_dict["private_classes"]
        private_classes.sort()
        n_classes = len(public_classes) + len(private_classes)

        N_parties = conf_dict["N_parties"]
        N_samples_per_class = conf_dict["N_samples_per_class"]

        N_rounds = conf_dict["N_rounds"]
        N_samples_per_class = conf_dict["N_samples_per_class"]

        N_private_training_round = conf_dict["N_private_training_round"]

        private_training_batchsize = conf_dict["private_training_batchsize"]

        N_logits_matching_round = conf_dict["N_logits_matching_round"]

        logits_matching_batchsize = conf_dict["logits_matching_batchsize"]

        # public dataset
        X_train_CIFAR10, y_train_CIFAR10, X_test_CIFAR10, y_test_CIFAR10 = load_CIFAR_data(data_type="CIFAR10", standardized=True, verbose=True)

        public_dataset = {"X": X_train_CIFAR10, "y": y_train_CIFAR10}

        # private dataset
        X_train_CIFAR100, y_train_CIFAR100, X_test_CIFAR100, y_test_CIFAR100 = load_CIFAR_data(data_type="CIFAR100", standardized=True, verbose=True)

        a_, y_train_super, b_, y_test_super = load_CIFAR_data(data_type="CIFAR100", label_mode="coarse", standardized=True, verbose=True)

        # Find the relations between superclasses and subclasses
        relations = [set() for i in range(np.max(y_train_super) + 1)]

        for i, y_fine in enumerate(y_train_CIFAR100):
            relations[y_train_super[i]].add(y_fine)

        for i in range(len(relations)):
            relations[i] = list(relations[i])


        fine_classes_in_use = [[relations[j][i%5] for j in private_classes] for i in range(N_parties)]     













            

        
    