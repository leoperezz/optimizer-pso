import torch.nn as nn
from .envs.mnist import MNISTDataset, MNISTModel
from .trainer import SupervisedTrainer, SupervisedTrainerPSO
from typing import Tuple

def load_env(env_name: str):
    if env_name == "mnist":
        dataset   = MNISTDataset()
        model     = MNISTModel()
        criterion = nn.CrossEntropyLoss()
        return dataset, model, criterion
    else:
        raise ValueError(f"Environment {env_name} not found")

def load_trainer(trainer_name: str,
                 model,
                 dataset,
                 criterion,
                 **kwargs):
    
    if trainer_name == "supervised":
        return SupervisedTrainer(model,
                                 dataset.get_train_loader(),
                                 dataset.get_test_loader(),
                                 criterion,
                                 **kwargs)
    
    elif trainer_name == "supervised-pso":
        return SupervisedTrainerPSO(model,
                                    dataset.get_train_loader(),
                                    dataset.get_test_loader(),
                                    criterion,
                                    **kwargs)
    else:
        raise ValueError(f"Trainer {trainer_name} not found")