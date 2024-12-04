import torch.nn as nn
from .envs.mnist import MNISTDataset, MNISTModel
from .trainer import SupervisedTrainer, SupervisedTrainerPSO

def load_env(env_name: str,
             batch_size: int = 128):
    if env_name == "mnist":
        dataset   = MNISTDataset(batch_size)
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
                                 optimizer           = kwargs["optimizer"],
                                 criterion           = criterion,
                                 device              = kwargs["device"],
                                 epochs              = kwargs["epochs"],
                                 learning_rate       = kwargs["learning_rate"])
    
    elif trainer_name == "supervised-pso":
        return SupervisedTrainerPSO(model,
                                    dataset.get_train_loader(),
                                    dataset.get_test_loader(),
                                    optimizer           = kwargs["optimizer"],
                                    criterion           = criterion,
                                    device              = kwargs["device"],
                                    epochs              = kwargs["epochs"],
                                    learning_rate       = kwargs["learning_rate"],
                                    num_particles       = kwargs["num_particles"],
                                    max_iter            = kwargs["max_iter"],
                                    lambda_factor       = kwargs["lambda_factor"],
                                    phi_lambda          = kwargs["phi_lambda"],
                                    phi_v               = kwargs["phi_v"],
                                    phi_p               = kwargs["phi_p"],
                                    phi_g               = kwargs["phi_g"],
                                    phi_w               = kwargs["phi_w"],
                                    c                   = kwargs["c"],
                                    c_r                 = kwargs["c_r"],
                                    K                   = kwargs["K"],
                                    perturbation_factor = kwargs["perturbation_factor"])
    else:
        raise ValueError(f"Trainer {trainer_name} not found")