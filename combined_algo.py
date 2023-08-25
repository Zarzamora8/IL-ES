import pandas as pd
import argparse
import os
# os.environ['RAY_OBJECT_STORE_MEMORY'] = '1'
# os.environ['RAY_HOME'] = './ray_cache'
# os.environ['TORCH_HOME'] = './ray_cache'
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
from encoder import Autoencoder
import random




def read_data(): 
    # reads expert data and outputs state and actions
    data = pd.read_pickle("./encoded_data/encoded_dataset.pkl")
    speed = np.stack(data['speed'], 0)
    X = np.stack(data["state"], 0)
    y = np.stack(data["action"], 0)
    X = np.hstack((X,speed[...,np.newaxis]))
    return X, y

    
def get_flat_weights(model):
        # Get the parameter tensors.
        theta_dict = model.state_dict()
        # Flatten it into a single np.ndarray.
        theta_list = []
        for k in sorted(theta_dict.keys()):
            theta_list.append(torch.reshape(theta_dict[k], (-1,)))
        cat = torch.cat(theta_list, dim=0)
        return cat.cpu().numpy()

def train(model, train_loader, criterion, optimizer, device):
    # train autoencoder 
    model.train()
    train_loss = 0
    for data in train_loader:
        inputs, label = data
        inputs = inputs.float().to(device)
        optimizer.zero_grad()
        outputs=[]
        for input in inputs:
            output = model(input)
            outputs.append(output)
        outputs = torch.stack(outputs)
        loss = criterion(outputs, label.long())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(train_loader)


@ray.remote(max_retries=0)
def train_population(args, training_set, num_inputs, num_outputs, nn_weights, device, i):
    # reward = 0
    # while True:
        # if reward == 0:
        #     pass
        # elif reward >=200:
        #     print("Sample:", i+1, "PASSSED!", " Reward:", reward)
        #     break
        # elif reward < 200:
        #     print("Sample:", i+1, "Needs to be retrained. ", " Reward:", reward)
    model = NeuralNetwork(num_inputs, num_outputs)
    # set_flat_weights(model, nn_weights)
    model.train()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001) # around 0.0001 - 0.00001 for batch size 32 (0.00008)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1, last_epoch=- 1, verbose=False)
    # Create data loader for the training set
    train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)

    for epoch in range(args.num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        if (epoch+1) % 50 == 0:
            print("Sample:", i+1, "Epoch:", epoch+1, " train loss:", train_loss)


    # ##########################################
    # env =  gym.make('CarRacing-v2', continuous=False, render_mode="human", lap_complete_percent= 1)
    # #################################################################

    # model.eval()
    # device = torch.device("cpu")
    # # run environment
    # reward = 0

    # obs, _ = env.reset()
    # reward = 0
    # for i in range(10000):
    #     obs = transforms.functional.crop(transform(obs), 0,6,84,84).to(device).float()
    #     obs = encoder.encoder(obs)
    #     speed = torch.from_numpy(env.true_speed[np.newaxis])
    #     obs = torch.cat((obs.flatten(), speed), 0)
    #     act = model(obs)
    #     # print(env.true_speed)
    #     # print(np.argmax(act.cpu().detach().numpy()))
    #     act = np.argmax(act.cpu().detach().numpy())
    #     obs, r, terminated, truncated, info= env.step(act)
    #     env.render()
    #     reward += r
    #     if terminated or truncated:
    #         print("Sample ", i, "reward= ", reward)
    #         break

    return get_flat_weights(model)

def set_flat_weights(model, theta):
    pos = 0
    theta_dict = model.state_dict()
    new_theta_dict = {}

    for k in sorted(theta_dict.keys()):
        shape = theta_dict[k].shape
        num_params = int(np.prod(shape))
        params_shape = np.reshape(theta[pos : pos + num_params], shape)
        new_theta_dict[k] = torch.from_numpy(params_shape.copy())
        pos += num_params
    model.load_state_dict(new_theta_dict)


@ray.remote
def test_population(env, model, weights, encoder, transform, device):
    # Make the population interact with the environment
    # Load the sample weights
    set_flat_weights(model, weights)
    # Create game loop
    obs, _ = env.reset()
    reward = 0
    for i in range(10000):
        obs = transforms.functional.crop(transform(obs), 0,6,84,84).to(device).float()
        obs = encoder.encoder(obs)
        speed = torch.from_numpy(env.true_speed[np.newaxis])
        obs = torch.cat((obs.flatten(), speed), 0)
        act = model(obs)
        # print(env.true_speed)
        # print(np.argmax(act.cpu().detach().numpy()))
        act = np.argmax(act.cpu().detach().numpy())
        obs, r, terminated, truncated, info= env.step(act)
        env.render()
        reward += r
        if terminated or truncated:
            print("Sample ", i, "reward= ", reward)
            break
    return reward
    



if __name__ == "__main__":
    writer = SummaryWriter()
    # Seeds 
    seed=1
    # Python random module.
    random.seed(seed)
    # Numpy.
    np.random.seed(seed)
    # Torch.
    torch.manual_seed(seed)

    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-p' , "--pop_size", type=int, default=50)
    parser.add_argument('-n' , "--num_epochs", type=int, default=20)
    parser.add_argument('-b' , "--batch_size", type=int, default=32)
    parser.add_argument('-e' , "--episodes", type=int, default=1)
    args = parser.parse_args()

    # torch device
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    ########################################################################################################################################
    #####################################################|||IMITATION LEARNING|||###########################################################
    ########################################################################################################################################

    # read data
    X, y = read_data()
    # Model hyperparameters
    num_inputs = X.shape[1]
    num_outputs = 5
    # split data
    training_set = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    # set a random torch seed
    torch.manual_seed(np.random.randint(0,10000))

    ##################################################||TEST IL||##################################################################

    # env =  gym.make('CarRacing-v2', continuous=False) #, render_mode="human")
    # load model 
    model = NeuralNetwork(num_inputs, num_outputs)
    model.to(device)
    nn_weights = get_flat_weights(model)
    # model.eval()

    # Ray parallel imitation learning computing
    # Set number of cpus and initialize ray
    num_cpus = 10
    ray.init(address="local", num_cpus=num_cpus)


    # ray.init(num_cpus=num_cpus)
    weights_table = []
    # parallel training of imitation learning neural networks
    result_refs = [train_population.remote(args, training_set, num_inputs, num_outputs, nn_weights, device, i) for i in range(args.pop_size+num_cpus-1)]
    # Check if any of the workers has finished and append weights of successful NN
    while len(result_refs):
        ready_refs, result_refs = ray.wait(result_refs)
        weights_table.append(ray.get(ready_refs[0]))
        if len(weights_table) == args.pop_size:
            break
    # Cancel the remaining workers that are still working after the population size has been reached
    for j in result_refs:
        ray.cancel(j)
    # Turn weights table into a dictionary
    weights_dict = dict(enumerate(weights_table, start=1))

    # env =  gym.make('CarRacing-v2', continuous=False, render_mode="human", lap_complete_percent= 1)
    # obj_ref = [test_population.remote(env, model, weights_dict[i+1], encoder, transform, device) for i in range(args.pop_size)] 
    # rewards_table = ray.get(obj_ref)
    # rewards_dict = dict(enumerate(rewards_table, start=1))
    # pprint(rewards_dict)
    



    ########################################################################################################################################
    #####################################################|||EVOLUTION STRATEGIES|||#########################################################
    ########################################################################################################################################

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




    for i in range(100):
        algo.train()
    
    path_to_checkpoint = algo.save()
    print(
    "An Algorithm checkpoint has been created inside directory: "
    f"'{path_to_checkpoint}'."
    )

    ########################################################################################################################################
    #####################################################|||RAY SHUTDOWN|||#################################################################
    ########################################################################################################################################