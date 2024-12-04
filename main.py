import argparse
import wandb
from optimzerpso.core.helpers import load_env, load_trainer

def get_args():
    parser = argparse.ArgumentParser(description='Training parameters for PSO-enhanced Neural Network')
    
    # Choose trainer
    parser.add_argument("--trainer", type=str, default="supervised",
                        choices=['supervised', 'supervised-pso'], help="Trainer type")

    # Dataset and basic training parameters
    parser.add_argument("--dataset", type=str, default="mnist",
                        help="Dataset to use for training")
    parser.add_argument("--optimizer", type=str, default="adam",
                        choices=['adam', 'sgd'], help="Optimizer type")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=['cuda', 'cpu'], help="Device to use for training")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate for optimizer")
    
    # PSO specific parameters
    parser.add_argument("--num_particles", type=int, default=20,
                        help="Number of particles in PSO")
    parser.add_argument("--max_iter", type=int, default=50,
                        help="Maximum iterations for PSO")
    parser.add_argument("--lambda_factor", type=float, default=0.95,
                        help="Step length for PSO")
    parser.add_argument("--phi_lambda", type=float, default=0.95,
                        help="Step length schedule factor")
    parser.add_argument("--phi_v", type=float, default=0.7,
                        help="Inertia weight")
    parser.add_argument("--phi_p", type=float, default=1.5,
                        help="Cognitive weight")
    parser.add_argument("--phi_g", type=float, default=1.5,
                        help="Social weight")
    parser.add_argument("--phi_w", type=float, default=2.0,
                        help="Repulsion weight")
    parser.add_argument("--c", type=int, default=10,
                        help="Patience for early stopping")
    parser.add_argument("--c_r", type=int, default=10,
                        help="Restart patience")
    parser.add_argument("--K", type=int, default=10,
                        help="PSO epochs")
    parser.add_argument("--perturbation_factor", type=float, default=0.3,
                        help="Factor for initial particle perturbation")
    
    return parser.parse_args()

def main():

    args = get_args()
    
    wandb.init(project="pso-neural-net", config=vars(args))

    dataset, model, criterion = load_env(args.dataset)
    
    trainer = load_trainer(args.trainer,
                           model,
                           dataset,
                           criterion,
                           **vars(args))
    
    trainer.train()

if __name__ == "__main__":
    main()