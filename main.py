import os
import yaml
import argparse

import numpy as np
import torch

from diffusion import Diffusion

torch.set_printoptions(sci_mode=False)


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def parse_args_and_config():
    parser = argparse.ArgumentParser(description="Argument parser for experiment")

    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--exp", type=str, default="exp", help="Path for saving running related data.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--test", action="store_true", help="Flag to test the model")
    parser.add_argument("--timesteps", type=int, default=1000, help="Number of steps involved")
    parser.add_argument("--eta", type=float, default=0.0, help="Control parameter for variance")
    parser.add_argument('-channels', default='3', help="Mode for model prediction: 1c (1 channel), 3c (3 channels)")

    args = parser.parse_args()
    args.log_path = os.path.join(args.exp, "logs")

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
        print(f"Directory '{args.log_path}' created successfully.")
    else:
        print(f"Directory '{args.log_path}' already exists.")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config_namespace = dict2namespace(config)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, config_namespace, device


def main():
    args, config, device = parse_args_and_config()
    print(f"config: {config}")
    runner = Diffusion(args, config, device)
    
    if args.test:
        print("test")
        runner.test()
    else:
        print("train")
        runner.train()

    return 0


if __name__ == "__main__":
    main()
