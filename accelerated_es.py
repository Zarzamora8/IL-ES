import pandas as pd
import argparse
import os
import gym
import glob
import numpy as np
from pprint import pprint
from sklearn.model_selection import train_test_split, KFold
import torch 
from torch import nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch.utils.data import  DataLoader,SubsetRandomSampler, TensorDataset
import torchvision.transforms as transforms
from ESTorchAlgo import ESConfig, ES, EvolutionNN, MyCallbacks
from ray.rllib.models import ModelCatalog
import ray
from run_model import NeuralNetwork
import logging

def get_flat_weights(model):
    # Get the parameter tensors.
    theta_dict = model.state_dict()
    # Flatten it into a single np.ndarray.
    theta_list = []
    for k in sorted(theta_dict.keys()):
        theta_list.append(torch.reshape(theta_dict[k], (-1,)))
    cat = torch.cat(theta_list, dim=0)
    return cat.cpu().numpy()


if __name__ == "__main__":
    writer = SummaryWriter()

    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-p' , "--pop_size", type=int, default=50)
    parser.add_argument('-n' , "--num_epochs", type=int, default=100)
    parser.add_argument('-b' , "--batch_size", type=int, default=32)
    parser.add_argument('-e' , "--episodes", type=int, default=1)
    args = parser.parse_args()

    # torch device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ########################################################################################################################################
    #####################################################|||IMITATION LEARNING|||###########################################################
    ########################################################################################################################################

    # set a random torch seed
    torch.manual_seed(1)

    # load model 
    logging.info("Loading IM model and creating population")
    model = NeuralNetwork(1025, 5)
    model.load_state_dict(torch.load("./model_weights/nn_weights.pt"))
    model.to(device)
    model.eval()
    weights = get_flat_weights(model)
    
    # Create population
    noise_table = np.random.normal(loc=0, scale=0.01, size=(len(weights), args.pop_size))
    weights_table = weights[:,np.newaxis] + noise_table
    weights_table[:,0] = weights
    # Turn weights table into a dictionary
    weights_dict = {idx+1: val for idx, val in enumerate(np.swapaxes(weights_table, 0, 1))}



    

    ########################################################################################################################################
    #####################################################|||EVOLUTION STRATEGIES|||#########################################################
    ########################################################################################################################################

    logging.info("Starting ES")
    ray.init(address='local', num_cpus=10)
    # Get model configuration
    config = ESConfig()
    # Register Model
    ModelCatalog.register_custom_model("torch_model", EvolutionNN)
    # Create an algorithm
    algo = ES(env="CarRacing-v2", config={
        "env_config" : {"continuous": False,
                        'lap_complete_percent': 1,
                        # 'render_mode':'human',
                        },
        "framework": "torch",
        "episodes_per_batch": args.episodes,
        "pop_size": args.pop_size,
        "noise_std": 1,
        "mutation": "normal",
        "sampler": "universal",
        "initial_pop": weights_dict,
        "num_rollout_workers": 10,
        "num_gpus": 0,
        "model": {  
        "custom_model": "torch_model",
        # "custom_model_config": {},
        },
    })




    for i in range(args.num_epochs):
        algo.train()
    
    path_to_checkpoint = algo.save()
    print(
    "An Algorithm checkpoint has been created inside directory: "
    f"'{path_to_checkpoint}'."
    )

    ########################################################################################################################################
    #####################################################|||RAY SHUTDOWN|||#################################################################
    ########################################################################################################################################