import torch
from torch import nn
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
import wandb


class Trainer(ABC):

    @abstractmethod
    def train_step(self):
        pass

    @abstractmethod
    def test_step(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass

    def load_optimizer(self, optimizer: str, learning_rate: float):
        if optimizer == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        else:
            raise ValueError("Optimizer must be either 'adam' or 'sgd'")


class SupervisedTrainer(Trainer):
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 optimizer: str,
                 criterion: nn.Module,
                 device: str,
                 epochs: int = 10,
                 learning_rate: float = 0.001):
        
        if optimizer not in ["adam", "sgd"]:
            raise ValueError("Optimizer must be either 'adam' or 'sgd'")

        self.model        = model
        self.train_loader = train_loader
        self.test_loader  = test_loader
        self.optimizer    = self.load_optimizer(optimizer, learning_rate)
        self.criterion    = criterion
        self.device       = device
        self.epochs       = epochs
        self.step_train   = 0
        self.step_test    = 0

    def train_step(self):
        
        self.model.train()

        for data, target in self.train_loader:
            
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)            
            loss   = self.criterion(output, target)
            wandb.log({"loss_train": loss.item()}, step=self.step_train)
            self.step_train += 1
            loss.backward()
            self.optimizer.step()

    def test_step(self):
        
        self.model.eval()
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss   = self.criterion(output, target)
                wandb.log({"loss_test": loss.item()}, step=self.step_test)
                self.step_test += 1
 
    def train(self):
        for _ in range(self.epochs):
            self.train_step()
            self.test_step()
            


class SupervisedTrainerPSO(Trainer):
    pass