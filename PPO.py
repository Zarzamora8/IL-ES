
import ray
import gym
import torchvision.transforms as transforms
import torch
from ray.tune.registry import register_env
from gym.envs.box2d.car_racing import CarRacing
import numpy as np
from encoder import Autoencoder
from ray.rllib.algorithms import ppo
from ray.rllib.utils.framework import try_import_torch
torch, nn = try_import_torch()
from ray.rllib.utils.annotations import override
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from typing import Dict, List, Optional
from gym.envs.box2d.car_racing import FrictionDetector
from gym.envs.box2d.car_dynamics import Car
import matplotlib.pyplot as plt

class MyCallbacks(DefaultCallbacks):
    
    @override(DefaultCallbacks)
    def on_train_result(
        self,
        *,
        algorithm: Optional["Algorithm"] = None,
        result: dict,
        trainer=None,
        **kwargs,
    ) -> None:
        """Called at the end of Trainable.train().
        Args:
            algorithm: Current trainer instance.
            result: Dict of results returned from trainer.train() call.
                You can mutate this object to add additional metrics.
            kwargs: Forward compatibility placeholder.
        """
        if trainer is not None:
            algorithm = trainer

        # if result['perf']['cpu_util_percent'] and result['perf']['gpu_util_percent0']:
        #     print(f"Iter: %.i;    mean_reward=%.2f,    max_reward=%.2f,   N_eff=%.2f,    episode_duration=%.2f s,   cpu_percent:%.2f,  gpu_percent:%.2f"\
        #      % (result['iterations_since_restore'], result['episode_reward_mean'], result['episode_reward_max'], result['n_eff'], result['episode_duration'],\
        #         result['perf']['cpu_util_percent'], result['perf']['gpu_util_percent0']))
        
        # elif result['perf']['gpu_util_percent0']:
        #     print(f"Iter: %.i;    mean_reward=%.2f,    max_reward=%.2f,   N_eff=%.2f,    episode_duration=%.2f s,    gpu_percent:%.2f"\
        #      % (result['iterations_since_restore'], result['episode_reward_mean'], result['episode_reward_max'], result['n_eff'], result['episode_duration'],\
        #         result['perf']['gpu_util_percent0']))

        elif result['perf']['cpu_util_percent']:
            print(f"Iter: %.i;    mean_reward=%.2f,    max_reward=%.2f,    episode_duration=%.2f s,   cpu_percent:%.2f"\
             % (result['iterations_since_restore'], result['episode_reward_mean'], result['episode_reward_max'],  result['time_this_iter_s'],
                result['perf']['cpu_util_percent']))

        else:
            print(f"Iter: %.i;    mean_reward=%.2f,    max_reward=%.2f,   N_eff=%.2f,    episode_duration=%.2f s"\
             % (result['iterations_since_restore'], result['episode_reward_mean'], result['episode_reward_max'], result['n_eff'], result['time_this_iter_s']))

class CustomCarRacing(gym.Env):
    def __init__(self, env_config):
        self.env = gym.make("CarRacing-v2", **env_config)
        self.action_space = self.env.action_space

    def reset(self):
        return self.env.reset(seed=1)

    def step(self, action):
        return self.env.step(action)

class ObsWrapper(gym.core.ObservationWrapper):
    def __init__(self, env, encoder):
        super().__init__(env)
        # Define new observation space
        self.encoder = encoder
        self.env = env
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(1025,), dtype=np.float32)
    # Override `observation` to custom process the original observation
    # coming from the env.
    def observation(self, observation):
        # Image transformation
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        obs = transforms.functional.crop(transform(observation), 0,6,84,84).to(device).float()
        # obs = obs.permute(1, 2, 0).to(device).float()
        obs = self.encoder(obs)
        speed = np.sqrt(np.square(self.env.env.car.hull.linearVelocity[0])+ np.square(self.env.env.car.hull.linearVelocity[1]))
        # speed = torch.from_numpy(self.env.true_speed[np.newaxis])
        obs = torch.cat((obs.flatten(), torch.from_numpy(speed[np.newaxis])), 0)
        # obs = torch.from_numpy(obs).float()
        return obs.detach().numpy()
    
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
#Subclass that inherits from pytorch's nn.Module and RlLib's TorchModelV2
class EvolutionNN(TorchModelV2, nn.Module):
    def __init__(
        self, observation_space, action_space, num_outputs, model_config, name
    ):
        TorchModelV2.__init__(
            self, observation_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        self.num_inputs = 1025
        # Neural Netwteork Structure
        self.net = nn.Sequential(
            nn.Linear(self.num_inputs, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, num_outputs),
        )

        self.net_val = nn.Sequential(
            nn.Linear(self.num_inputs, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
    # Forward Pass    
    def forward(self, input_dict, state, seq_lens):
        self.input = input_dict["obs_flat"]
        return self.net(input_dict["obs_flat"]), []


    def value_function(self):
        # Forward pass for value function
        return self.net_val(self.input[0])

    
if __name__ == "__main__":
    # --------------------Encoder-------------------------
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    # Use the loaded encoder
    encoder = Autoencoder().to(device)
    encoder.encoder = torch.load('./model_weights/good_encoder.pth', map_location="cpu")
    # --------------------Encoder-------------------------

    ray.init(address='local', num_cpus=10, num_gpus=0)   
    def create_grayscale_car_racing_env(env_config):
        encoder = Autoencoder().to(device)
        encoder.encoder = torch.load('./model_weights/good_encoder.pth', map_location="cpu")
        env = ObsWrapper(CustomCarRacing(env_config), encoder.encoder)
        return env
    
    pretrained_weights = './model_weights/nn_weights.pt'
    model = EvolutionNN(1025,5)
    model.load_state_dict(torch.load('./model_weights/nn_weights.pt'))   
    flat_weights = torch.cat([param.data.flatten() for param in model.parameters()])

    register_env("grayscale_car_racing", create_grayscale_car_racing_env)
    ModelCatalog.register_custom_model("torch_model", EvolutionNN)
    # Build a Algorithm object from the config and run 1 training iteration.
    algo = ppo.PPO(env="grayscale_car_racing", config={
    'disable_env_checking' : True,
    "callbacks" : MyCallbacks,
    "env_config": {
        "continuous": False,
        'lap_complete_percent': 1,
                    }, # config to pass to env class
    'seed' : 1,
    "framework": "torch",
    # 'lr_schedule' : [[0, 5e-5], [500_000, 3e-5], [1_000_000, 1e-5], [2_000_000, 5e-6], [3_000_000, 5e-7]],
    'lr' : 5e-3,
    "train_batch_size": 10_000,
    'sgd_minibatch_size' : 128,
    "num_rollout_workers": 10,
    "num_sgd_iter" : 30,
    "num_gpus": 0,
    "model": {  
    "custom_model": "torch_model",
    },
})
    
    policy = algo.get_policy()
    idx=0
    for param in policy.model.parameters():
        param.data = flat_weights[idx:idx + param.numel()].reshape(param.shape)
        idx += param.numel()


    for _ in range(400):
        algo.train()

    path_to_checkpoint = algo.save()
    print(
    "An Algorithm checkpoint has been created inside directory: "
    f"'{path_to_checkpoint}'."
    )