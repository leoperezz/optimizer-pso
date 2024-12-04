import torch
from torch import nn
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
import wandb
from tqdm import tqdm
from .logger import logger
from typing import List, Tuple

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

    def load_optimizer(self, optimizer: str, learning_rate: float):
        if optimizer == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        else:
            raise ValueError("Optimizer must be either 'adam' or 'sgd'")


class SupervisedTrainer(Trainer):
    
    def __init__(self,
                 model:         nn.Module,
                 train_loader:  DataLoader,
                 test_loader:   DataLoader,
                 optimizer:     str,
                 criterion:     nn.Module,
                 device:        str,
                 epochs:        int = 10,
                 learning_rate: float = 1e-3):
        
        if optimizer not in ["adam", "sgd"]:
            raise ValueError("Optimizer must be either 'adam' or 'sgd'")

        self.model        = model.to(device)
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
            wandb.log({"train/loss": loss.item(), "train/step": self.step_train})
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
                wandb.log({"test/loss": loss.item(), "test/step": self.step_test})
                self.step_test += 1
 
    def train(self):
        
        wandb.define_metric("train/loss", step_metric="train/step")
        wandb.define_metric("test/loss", step_metric="test/step")

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
                 model:         nn.Module,
                 train_loader:  DataLoader,
                 test_loader:   DataLoader,
                 optimizer:     str,
                 criterion:     nn.Module,
                 device:        str,
                 epochs:        int = 10,
                 learning_rate: float = 1e-3,
                 num_particles: int = 20,
                 max_iter:      int = 50,
                 lambda_factor: float = 0.95, #step_length
                 phi_lambda:    float = 0.95, #step_length_schedule
                 phi_v:         float = 0.7,  #inertia_weight
                 phi_p:         float = 1.5,  #cognitive_weight
                 phi_g:         float = 1.5,  #social_weight
                 phi_w:         float = 2.0,  #repulsion_weight
                 c:             int   = 10,   #patience
                 c_r:           int   = 10,   #restart_patience
                 K:             int   = 10,   #pso_epochs
                 perturbation_factor: float = 0.3):
        
        self.model               = model.to(device)
        self.train_loader        = train_loader
        self.test_loader         = test_loader
        self.optimizer           = self.load_optimizer(optimizer, learning_rate)
        self.criterion           = criterion
        self.device              = device
        self.epochs              = epochs
        self.step_train          = 0
        self.step_test           = 0
        self.num_particles       = num_particles
        self.max_iter            = max_iter
        self.K                   = K
        self.interval_pso        = len(self.train_loader) // K
        self.perturbation_factor = perturbation_factor
        self.lambda_factor       = lambda_factor
        self.phi_lambda          = phi_lambda
        self.phi_v               = phi_v
        self.phi_p               = phi_p
        self.phi_g               = phi_g
        self.phi_w               = phi_w
        self.c                   = c
        self.c_r                 = c_r


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
            # Crear una nueva instancia del modelo con la misma arquitectura
            new_particle = type(self.model)().to(self.device)
            new_particle.load_state_dict(self.model.state_dict())
            
            with torch.no_grad():
                for param in new_particle.parameters():
                    std = self.perturbation_factor * torch.std(param.data)
                    perturbation = torch.randn_like(param.data) * std
                    param.data += perturbation
            
            particles.append(new_particle)
        
        return particles
 
    def pso(self, batch):
        """
        Particle Swarm Optimization for neural network weights
        Args:
            batch: Tuple[torch.Tensor, torch.Tensor] - (data, target) for evaluation
        Returns:
            None - Updates self.model directly
        Notes:
            * Falta corregir la parte en la que se actualiza el unchanged_counter_global y el unchanged_counter_local
        """
        data, target = batch
        data, target = data.to(self.device), target.to(self.device)
        
        # Initialize particles
        particles = self.initialize_particles()
        velocities = []
        personal_best = []
        personal_best_scores = []
        
        # Initialize velocities and personal bests
        for particle in particles:
            particle = particle.to(self.device)
            velocity = {name: torch.zeros_like(param, device=self.device) 
                       for name, param in particle.named_parameters()}
            velocities.append(velocity)
            personal_best.append(particle.state_dict())
            
            # Evaluate initial position (negative loss as we want to maximize)
            with torch.no_grad():
                output = particle(data)
                score = -self.criterion(output, target).item()
                personal_best_scores.append(score)
        
        # Initialize global best
        global_best_idx = max(range(len(personal_best_scores)), 
                             key=lambda i: personal_best_scores[i])
        global_best = particles[global_best_idx].state_dict()
        global_best_score = personal_best_scores[global_best_idx]
        
        # Initialize global worst
        global_worst_idx = min(range(len(personal_best_scores)), 
                              key=lambda i: personal_best_scores[i])
        global_worst = particles[global_worst_idx].state_dict()
        global_worst_score = personal_best_scores[global_worst_idx]
        
        # Counter for early stopping
        unchanged_counter_global = 0
        
        # Main PSO loop
        for k in tqdm(range(self.max_iter), desc="PSO", total=self.max_iter):
            
            logger.info(f"PSO iteration: {k}, unchanged_counter: {unchanged_counter_global}")
            
            if unchanged_counter_global >= self.c:
                logger.exito(f"Early stopping: {unchanged_counter_global} >= {self.c}")
                break
            
            unchanged_counter_local = 0

            for i in range(self.num_particles):
                '''
                This must be in parallel
                '''
                particle = particles[i]
                
                # Generate random factors
                r_v = torch.rand(1).item()
                r_p = torch.rand(1).item()
                r_g = torch.rand(1).item()
                r_w = torch.rand(1).item()
                
                # Normalization term
                C = (r_v * self.phi_v + r_p * self.phi_p + r_g * self.phi_g + r_w * self.phi_w)
                
                # Update velocity and position for each parameter
                with torch.no_grad():
                    for name, param in particle.named_parameters():
                        # Update velocity
                        velocities[i][name] = (1/C) * (
                            r_v * self.phi_v * velocities[i][name] +
                            r_p * self.phi_p * (personal_best[i][name].to(self.device) - param) +
                            r_g * self.phi_g * (global_best[name].to(self.device) - param) -
                            r_w * self.phi_w * (global_worst[name].to(self.device) - param)
                        )
                        
                        # Update position
                        param.data += self.lambda_factor * velocities[i][name]
                
                # Evaluate new position
                with torch.no_grad():
                    output = particle(data)
                    score = -self.criterion(output, target).item()
                
                # Update personal best
                if score > personal_best_scores[i]:
                    personal_best[i] = particle.state_dict()
                    personal_best_scores[i] = score
                else:
                    unchanged_counter_local += 1
                if unchanged_counter_local >= self.c_r:
                    particle.load_state_dict(personal_best[i])
                    velocities[i] = torch.zeros_like(velocities[i])
                    break
                
                # Update global best and worst
                if score > global_best_score:
                    new_global_best_idx = i
                    global_best = particle.state_dict()
                    global_best_score = score
                
                if score < global_worst_score:
                    global_worst = particle.state_dict()
                    global_worst_score = score
                
            
            if new_global_best_idx != global_best_idx:
                global_best_idx = new_global_best_idx
            else:
                unchanged_counter_global += 1
            
            # Step length scheduling
            self.lambda_factor *= self.phi_lambda
        
        self.model.load_state_dict(global_best)
        
        if hasattr(self.optimizer, 'state'):
            optimizer_state = self.optimizer.state_dict()
            if isinstance(self.optimizer, torch.optim.Adam):
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=optimizer_state['param_groups'][0]['lr'])
            elif isinstance(self.optimizer, torch.optim.SGD):
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=optimizer_state['param_groups'][0]['lr'])
            self.optimizer.load_state_dict(optimizer_state)
        
        # Clean up memory
        del particles, velocities, personal_best, personal_best_scores
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return None

    def train_step(self):
        logger.worker("Training")
        self.model.train()

        for data, target in tqdm(self.train_loader,
                                desc="Training",
                                total=len(self.train_loader),
                                colour="green"):
            
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)            
            loss = self.criterion(output, target)
            wandb.log({"loss_train": loss.item()}, step=self.step_train)
            self.step_train += 1
            loss.backward()
            self.optimizer.step()

            if self.step_train % self.interval_pso == 0:
                batch = next(iter(self.train_loader))
                logger.exito("Making PSO ...")
                self.pso(batch)

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
