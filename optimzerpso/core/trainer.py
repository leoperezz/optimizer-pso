import torch
from torch import nn
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
import wandb
from tqdm import tqdm
from logger import logger

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
                 learning_rate: float = 1e-3):
        
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

        logger.worker("Training")
        
        self.model.train()

        for data, target in tqdm(self.train_loader,
                                 desc     = "Training",
                                 total    = len(self.train_loader),
                                 colour   = "green"):
            
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)            
            loss   = self.criterion(output, target)
            wandb.log({"loss_train": loss.item()}, step=self.step_train)
            self.step_train += 1
            loss.backward()
            self.optimizer.step()

    def test_step(self):

        logger.worker("Testing")
        
        self.model.eval()
        
        with torch.no_grad():
            for data, target in tqdm(self.test_loader,
                                     desc     = "Testing",
                                     total    = len(self.test_loader),
                                     colour   = "blue"):
                
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss   = self.criterion(output, target)
                wandb.log({"loss_test": loss.item()}, step=self.step_test)
                self.step_test += 1
 
    def train(self):
        for epoch in range(self.epochs):
            logger.worker(f"Epoch {epoch + 1} of {self.epochs}")
            self.train_step()
            self.test_step()



class SupervisedTrainerPSO(Trainer):
    
    '''
    Tal vez uno de los problemas que tengo es que no se como hacer para que el PSO
    pueda funcionar con la red neuronal. Los problemas son:
    * Memoria
    * Convergencia Lenta
    * Cuanto debe de ser el K (PSO x Epochs)
    * Como definimos la función f: R_{theta} -> R (tal vez usando un batch size o sampleando algo específico)
    '''

    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 optimizer: str,
                 criterion: nn.Module,
                 device: str,
                 epochs: int = 10,
                 learning_rate: float = 1e-3,
                 num_particles: int = 10,
                 max_iter: int = 100,
                 step_length: float = 1.0,
                 step_length_schedule: float = 0.9,
                 inertia_weight: float = 0.7,
                 cognitive_weight: float = 1.5,
                 social_weight: float = 1.5,
                 repulsion_weight: float = 2.0,
                 patience: int = 10,
                 K: int = 10,
                 perturbation_factor: float = 0.3):
        
        super().__init__(model, train_loader, test_loader, optimizer, criterion, device, epochs, learning_rate)

        self.num_particles       = num_particles
        self.max_iter            = max_iter
        self.K                   = K
        self.interval_pso        = len(self.train_loader) // K
        self.perturbation_factor = perturbation_factor
        self.patience            = patience
        self.step_length         = step_length
        self.step_length_schedule = step_length_schedule
        self.inertia_weight      = inertia_weight
        self.cognitive_weight    = cognitive_weight
        self.social_weight       = social_weight
        self.repulsion_weight    = repulsion_weight


    def initialize_particles(self):
        """
        Initialize PSO particles based on the current model parameters
        
        The first particle is the current model, and the rest are new instances of the same model architecture
        with perturbed parameters.

        The perturbation factor is a constant value that is used to scale the standard deviation of the perturbation. 
        Controls how much the parameters are perturbed. Is the factor that controls exploration vs exploitation.

        """
        particles = [self.model]  

        for _ in range(self.num_particles - 1):
            # Create a new instance of the same model architecture
            new_particle = type(self.model)(*self.model.__init__args__).to('cpu')
            new_particle.load_state_dict(self.model.state_dict())
            
            with torch.no_grad():
                for param in new_particle.parameters():

                    std = self.perturbation_factor * torch.std(param.data)
                    
                    perturbation = torch.randn_like(param.data) * std

                    param.data += perturbation
            
            particles.append(new_particle)
        
        return particles

    def pso(self):
        
        particles = self.initialize_particles()

    def train_step(self):

        logger.worker("Training")
        
        self.model.train()

        for data, target in tqdm(self.train_loader,
                                 desc     = "Training",
                                 total    = len(self.train_loader),
                                 colour   = "green"):
            
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)            
            loss   = self.criterion(output, target)
            wandb.log({"loss_train": loss.item()}, step=self.step_train)
            self.step_train += 1
            loss.backward()
            self.optimizer.step()

            if self.step_train % self.interval_pso == 0:
                self.pso()

    def test_step(self):

        logger.worker("Testing")
        
        self.model.eval()
        
        with torch.no_grad():
            for data, target in tqdm(self.test_loader,
                                     desc     = "Testing",
                                     total    = len(self.test_loader),
                                     colour   = "blue"):
                
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss   = self.criterion(output, target)
                wandb.log({"loss_test": loss.item()}, step=self.step_test)
                self.step_test += 1

    def train(self):
        pass
