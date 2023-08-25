from collections import namedtuple, Counter, defaultdict, OrderedDict
import logging
import numpy as np
import pandas as pd
import os
import random
import time
import gym
from typing import Dict, List, Optional
from encoder import Autoencoder
import torchvision.transforms as transforms
import pprint
from scipy.linalg import ldl, sqrtm
import ray
from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.es import optimizers, utils
from ray.rllib.algorithms.es.es_tf_policy import ESTFPolicy, rollout
from ray.rllib.env.env_context import EnvContext
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils import FilterManager
from ray.rllib.utils.actor_manager import FaultAwareApply
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import Deprecated
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_AGENT_STEPS_TRAINED,
    NUM_ENV_STEPS_SAMPLED,
    NUM_ENV_STEPS_TRAINED,
)
from ray.rllib.utils.torch_utils import set_torch_seed
from ray.rllib.utils.typing import AlgorithmConfigDict, PolicyID
from run_model import NeuralNetwork

logger = logging.getLogger(__name__)

from ray.rllib.utils.framework import try_import_torch
torch, nn = try_import_torch()

from ray.rllib.algorithms.callbacks import DefaultCallbacks

# Seeds 
seed=3
# Python random module.
random.seed(seed)
# Numpy.
np.random.seed(seed)
# Torch.
set_torch_seed(seed)

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
            print(f"Iter: %.i;    mean_reward=%.2f,    max_reward=%.2f,   noise_std=%.4f,   N_eff=%.2f,    episode_duration=%.2f s,   cpu_percent:%.2f"\
             % (result['iterations_since_restore'], result['episode_reward_mean'], result['episode_reward_max'], result['noise_std'], result['n_eff'], result['episode_duration'],\
                result['perf']['cpu_util_percent']))

        else:
            print(f"Iter: %.i;    mean_reward=%.2f,    max_reward=%.2f,   N_eff=%.2f,    episode_duration=%.2f s"\
             % (result['iterations_since_restore'], result['episode_reward_mean'], result['episode_reward_max'], result['n_eff'], result['episode_duration']))


class ESConfig(AlgorithmConfig):
    """Defines a configuration class from which an ES Algorithm can be built.
    Example:
        >>> from ray.rllib.algorithms.es import ESConfig
        >>> config = ESConfig().training(sgd_stepsize=0.02, report_length=20)\
        ...     .resources(num_gpus=0)\
        ...     .rollouts(num_rollout_workers=4)
        >>> print(config.to_dict())
        >>> # Build a Algorithm object from the config and run 1 training iteration.
        >>> trainer = config.build(env="CartPole-v1")
        >>> trainer.train()
    Example:
        >>> from ray.rllib.algorithms.es import ESConfig
        >>> from ray import tune
        >>> config = ESConfig()
        >>> # Print out some default values.
        >>> print(config.action_noise_std)
        >>> # Update the config object.
        >>> config.training(rollouts_used=tune.grid_search([32, 64]), eval_prob=0.5)
        >>> # Set the config object's env.
        >>> config.environment(env="CartPole-v1")
        >>> # Use to_dict() to get the old-style python config dict
        >>> # when running with tune.
        >>> tune.Tuner(
        ...     "ES",
        ...     run_config=ray.air.RunConfig(stop={"episode_reward_mean": 200}),
        ...     param_space=config.to_dict(),
        ... ).fit()
    """

    def __init__(self):
        """Initializes a ESConfig instance."""
        super().__init__(algo_class=ES)

        # fmt: off
        # __sphinx_doc_begin__

        # ES specific settings:
        self.pop_size = 100
        self.noise_std = 0.01
        self.episodes_per_batch = 10
        self.report_length = 10
        self.mutation = "normal"
        self.sampler = "universal"
        self.initial_pop = None
        # Override some of AlgorithmConfig's default values with ES-specific values.
        self.train_batch_size = 1
        # self.num_workers = 10
        self.callbacks = MyCallbacks
        self.observation_filter = "NoFilter"
        # self.evaluation(
        #     evaluation_config=AlgorithmConfig.overrides(
        #         num_envs_per_worker=1,
        #         observation_filter="NoFilter",
        #     )
        # )

        # __sphinx_doc_end__
        # fmt: on

    @override(AlgorithmConfig)
    def training(
        self,
        *,
        pop_size: Optional[int] = None,
        noise_std: Optional[float] = None,
        episodes_per_batch: Optional[int] = None,
        report_length: Optional[int] = None,
        sampler: Optional[str] = None,
        mutation: Optional[str] = None,
        initial_pop: Optional[dict] = None,
        **kwargs,
    ) -> "ESConfig":
        """Sets the training related configuration.
        Args:
            action_noise_std: Std. deviation to be used when adding (standard normal)
                noise to computed actions. Action noise is only added, if
                `compute_actions` is called with the `add_noise` arg set to True.
            l2_coeff: Coefficient to multiply current weights with inside the globalg
                optimizer update term.
            noise_stdev: Std. deviation of parameter noise.
            episodes_per_batch: Minimum number of episodes to pack into the train batch.
            eval_prob: Probability of evaluating the parameter rewards.
            stepsize: SGD step-size used for the Adam optimizer.
            noise_size: Number of rows in the noise table (shared across workers).
                Each row contains a gaussian noise value for each model parameter.
            report_length: How many of the last rewards we average over.
        Returns:
            This updated AlgorithmConfig object.
        """
        # Pass kwargs onto super's `training()` method.
        super().training(**kwargs)
        
        if pop_size is not None:
            self.pop_size = pop_size
        if noise_std is not None:
            self.noise_std = noise_std
        if episodes_per_batch is not None:
            self.episodes_per_batch = episodes_per_batch
        if report_length is not None:
            self.report_length = report_length
        if mutation is not None:
            self.mutation = mutation
        if sampler is not None:
            self.sampler = sampler
        if initial_pop is not None:
            self.initial_pop = initial_pop
        return self

    @override(Algorithm)
    def validate_config(self, config: AlgorithmConfigDict) -> None:
        # Call super's validation method.
        super().validate_config(config)

        if config["num_gpus"] > 1:
            raise ValueError("`num_gpus` > 1 not yet supported for ES!")
        if self.num_rollout_workers <= 0:
            raise ValueError("`num_rollout_workers` must be > 0 for ES!")
        if config["evaluation_config"]["num_envs_per_worker"] != 1:
            raise ValueError(
                "`evaluation_config.num_envs_per_worker` must always be 1 for "
                "ES! To parallelize evaluation, increase "
                "`evaluation_num_workers` to > 1."
            )
        if config["evaluation_config"]["observation_filter"] != "NoFilter":
            raise ValueError(
                "`evaluation_config.observation_filter` must always be "
                "`NoFilter` for ES!"
            )
    
# def get_flat_weights(policy):
#         # Get the parameter tensors.
#         theta_dict = policy.model.state_dict()
#         # Flatten it into a single np.ndarray.
#         theta_list = []
#         for k in sorted(theta_dict.keys()):
#             theta_list.append(torch.reshape(theta_dict[k], (-1,)))
#         cat = torch.cat(theta_list, dim=0)
#         return cat.cpu().numpy()
def get_flat_weights(model):
        # Get the parameter tensors.
        theta_dict = model.state_dict()
        # Flatten it into a single np.ndarray.
        theta_list = []
        for k in sorted(theta_dict.keys()):
            theta_list.append(torch.reshape(theta_dict[k], (-1,)))
        cat = torch.cat(theta_list, dim=0)
        return cat.cpu().numpy()
    
def create_weights_dict(policy, pop_size, noise_std):
    model = NeuralNetwork(1025,5)
    w_flat = get_flat_weights(model)
    noise_table = np.random.normal(loc=0, scale=noise_std, size=(len(w_flat), pop_size))
    weights_dict = {idx + 1 : w_flat + noise_table[:,idx] for idx in range(pop_size)}
    return weights_dict


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
        speed = np.sqrt(np.square(self.env.car.hull.linearVelocity[0])+ np.square(self.env.car.hull.linearVelocity[1]))
        # speed = torch.from_numpy(self.env.true_speed[np.newaxis])
        obs = torch.cat((obs.flatten(), torch.from_numpy(speed[np.newaxis])), 0)
        # obs = torch.from_numpy(obs).float()
        return obs.detach().numpy()


@ray.remote
class Worker:
    def __init__(
        self,
        config: AlgorithmConfig,
        policy_params,
        env,
        worker_index,
        min_task_runtime=0.2,
    ):

        # Set Python random, numpy, env, and torch/tf seeds.
        seed = config.get("seed")
        if seed is not None:
            # Python random module.
            random.seed(seed)
            # Numpy.
            np.random.seed(seed)
            # Torch.
            if config.get("framework") == "torch":
                set_torch_seed(seed)
        self.seed = seed
        self.min_task_runtime = min_task_runtime
        self.config = config
        self.config.update_from_dict(policy_params)
        # self.config["single_threaded"] = True

        # env_context = EnvContext(config["env_config"] or {}, worker_index)
        # self.env = env_creator(env_context)
        self.env = env
        from ray.rllib import models

        # self.preprocessor = models.ModelCatalog.get_preprocessor(
        #     self.env, config["model"]
        # )

        _policy_class = get_policy_class(config)
        self.policy = _policy_class(
            self.env.observation_space, self.env.action_space, config
        )
        # print(self.env.observation_space.shape, self.env.action_space)
        self.model = NeuralNetwork(1025,5)
        self.model.eval()

    @property
    def filters(self):
        return {DEFAULT_POLICY_ID: self.policy.observation_filter}

    def sync_filters(self, new_filters):
        for k in self.filters:
            self.filters[k].sync(new_filters[k])

    def get_filters(self, flush_after=False):
        return_filters = {}
        for k, f in self.filters.items():
            return_filters[k] = f.as_serializable()
            if flush_after:
                f.reset_buffer()
        return return_filters

    def rollout(self, env_seed, timestep_limit: Optional[int] = None):
        # Compute a simulation episode.
        rewards = []
        t = 0
        max_timestep_limit = 999999
        env_timestep_limit = (self.env.spec.max_episode_steps if (hasattr(self.env, "spec") and hasattr(self.env.spec, "max_episode_steps")) else max_timestep_limit)
        timestep_limit = (env_timestep_limit if timestep_limit is None else min(timestep_limit, env_timestep_limit))
        # obs, _ = self.env.reset(seed=env_seed)
        for i in range(self.config["episodes_per_batch"]):
            obs, _ = self.env.reset(seed=1)
            # obs, _ = self.env.reset()
            reward = 0
            for _ in range(timestep_limit or max_timestep_limit): 
                act = self.model(torch.from_numpy(obs))
                act = np.argmax(act.cpu().detach().numpy())
                # act, _, __ = self.policy.compute_single_action(obs)
                obs, r, terminated, truncated, info = self.env.step(act)
                reward += r 
                t +=1
                # self.env.render()
                if terminated or truncated:
                    break
            rewards.append(reward)
        rewards = np.mean(rewards)
        return rewards, t

    def do_rollouts(self, params, env_seed, timestep_limit = None):
        # Set the network weights.
        self.policy.set_flat_weights(self.model, params)
        rewards, length = self.rollout(env_seed)
        result = (rewards, length)
        return result
        


def get_policy_class(config):
    if config["framework"] == "torch":
        from ESTorchPolicy import ESTorchPolicy

        policy_cls = ESTorchPolicy
    else:
        policy_cls = ESTFPolicy
    return policy_cls

# @ray.remote(num_gpus=2)
# def particle_filter(matrix, aweights):
#     warnings.filterwarnings("ignore")
#     matrix = matrix.T
#     matrix_gpu = torch.from_numpy(matrix).to(device)
#     aweights_gpu = torch.from_numpy(aweights).to(device)
#     torch.cuda.synchronize()
#     cov_matrix = torch.cov(matrix_gpu, correction=0, aweights=aweights_gpu)
#     torch.cuda.synchronize()
#     return cov_matrix


class ES(Algorithm):
    """Large-scale implementation of Evolution Strategies in Ray."""

    @classmethod
    @override(Algorithm)
    def get_default_config(cls) -> AlgorithmConfig:
        return ESConfig()


    @override(Algorithm)
    def setup(self, config):
        # Setup our config: Merge the user-supplied config (which could
        # be a partial config dict with the class' default).
        if isinstance(config, dict):
            self.config = self.get_default_config().update_from_dict(config)

        # Call super's validation method.
        self.config.validate()

        # Generate the local env.
        env_context = EnvContext(self.config["env_config"] or {}, worker_index=0)
        env = self.env_creator(env_context)
        # --------------------Encoder-------------------------
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        # Use the loaded encoder
        encoder = Autoencoder().to(device)
        encoder.encoder = torch.load('./model_weights/good_encoder.pth', map_location="cpu")
        # --------------------Encoder-------------------------
        self.env = ObsWrapper(env, encoder.encoder)
        self.callbacks = MyCallbacks()
        self.pop_size = self.config["pop_size"]
        self.noise_std = self.config["noise_std"]
            
        self._policy_class = get_policy_class(self.config)
        self.policy = self._policy_class(
            obs_space=self.env.observation_space,
            action_space=self.env.action_space,
            config=self.config,
        )
        self.report_length = self.config["report_length"]

        # Create weights table
        logger.info("Creating population.")
        if self.config["initial_pop"]:
            self.weights_dict = self.config["initial_pop"]
        else:
            self.weights_dict = create_weights_dict(self.policy, self.pop_size, self.noise_std)

        # algorithm specific arguments
        if self.config["mutation"] == "particle_filter":
            self.w_all = {i : 1/self.pop_size for i in range(1,self.pop_size+1)}
        elif self.config["mutation"] == "normal":
            self.last_mean = self.weights_dict[1]
        elif self.config["mutation"] == "covariance":
            # mean = np.zeros((10427))
            # variance = np.zeros((10427))
            mean = np.zeros((16775))
            variance = np.zeros((16775))
            for i in range(1, self.pop_size+1):
                    mean +=  self.weights_dict[i] 
            mean /= self.pop_size
            for i in range(1, self.pop_size+1):        
                    yi = ((self.weights_dict[i] -  mean)**2)
                    variance += yi
            noise = np.sqrt(variance/self.pop_size)
            self.last_cov = np.multiply(noise, np.identity(len(self.weights_dict[1])))
            self.max = 0
        elif self.config["mutation"] == "cma":
            mean = np.zeros((10427))
            variance = np.zeros((10427))
            for i in range(1, self.pop_size+1):
                    mean +=  self.weights_dict[i] 
            mean /= self.pop_size
            for i in range(1, self.pop_size+1):        
                    yi = ((self.weights_dict[i] -  mean)**2)
                    variance += yi
            noise = np.sqrt(variance/self.pop_size)
            self.last_mean = mean
            self.last_cov = np.multiply(noise, np.identity(len(self.weights_dict[1])))
            self.pc = 0
            self.po = 0
            self.low, d, _ = ldl(self.last_cov)
            self.d = np.sqrt(np.diagonal(d))
        # Create the actors.
        logger.info("Creating actors.")
        self.workers = [
            Worker.remote(self.config, {}, self.env, idx + 1)
            for idx in range(self.config.num_rollout_workers)
        ]

        self.episodes_so_far = 0
        self.reward_list = []
        self.tstart = time.time()

    @override(Algorithm)
    def get_policy(self, policy=DEFAULT_POLICY_ID):
        if policy != DEFAULT_POLICY_ID:
            raise ValueError(
                "ES has no policy '{}'! Use {} "
                "instead.".format(policy, DEFAULT_POLICY_ID)
            )
        return self.policy
    
    @override(Algorithm)
    def step(self):
        # perform a step in the training
        config = self.config
       # Collect results from rollouts
        fitness, lengths, num_episodes, num_timesteps = self._collect_results()
        self.save_results(fitness)
        # Update our sample steps counters.
        self._counters[NUM_AGENT_STEPS_SAMPLED] += num_timesteps
        self._counters[NUM_ENV_STEPS_SAMPLED] += num_timesteps
        self.episodes_so_far += num_episodes
        # Assemble the results.
        fitness = np.array(fitness)
        # print(fitness)
        mean_fitness = np.mean(fitness)
        max_fitness = np.max(fitness)
        std_fitness = np.std(fitness)
        eval_lengths = np.array(lengths)
        # Mutation Selection
        if self.config["mutation"] == "normal":
            self.mutate(fitness)
            self.noise_adaptation(mean_fitness)
        elif self.config["mutation"] == "particle_filter":
            # Resampler Selection
            if self.config["sampler"] == "universal":
                survivors = self.uni_sampler(fitness)
            elif self.config["sampler"] == "residual":
                survivors = self.res_sampler(fitness)
            else:
                raise Exception("Resampler not supported")
            self.particle_filter_sampling(survivors)
        elif self.config["mutation"] == "covariance":
            self.accelerated_es(fitness)
        elif self.config["mutation"] == "cma":
            self.cma(fitness)
        else:
            raise Exception("Mutation model not supported") 
        # Store the rewards
        self.reward_list.append(mean_fitness)
        # Calculate N-eff
        norm_fitness = fitness/np.sum(fitness)
        N_eff = 1/(np.sum(norm_fitness**2))
        # Define Callbacks to be return at the end of trainer.train()
        info = {
            "episodes_so_far": self.episodes_so_far,
        }

        result = dict(
            episode_reward_mean=mean_fitness,
            episode_reward_std=std_fitness,
            episode_reward_max=max_fitness,
            n_eff=N_eff,
            pop_size = self.pop_size,
            noise_std=self.noise_std,
            episode_duration=round(time.time()-self.start, 2),
            episode_len_mean=eval_lengths.mean(),
            info=info,
        )

        return result
    
    def save_results(self, fitness):
        new_df = pd.DataFrame({str(i+1): fit for i,fit in enumerate(fitness)}, index=[0])
        folder_path = "C:/Users/Z0159590/ImitationLearning Ray 22/final_results/IL_ES_pop50"
        file ="population_fitness_diff.csv"
        file_path = os.path.join(folder_path, file)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print("Directory '%s' created" %file_path)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, index_col=0)
            df = pd.concat([df, new_df], ignore_index=True)
            df.to_csv(file_path)
        else:
            new_df.to_csv(file_path)


    
    @override(Algorithm)
    def compute_single_action(self, observation, *args, **kwargs):
        action, _, _ = self.policy.compute_actions([observation], update=False)
        if kwargs.get("full_fetch"):
            return action[0], [], {}
        return action[0]

    @override(Algorithm)
    def cleanup(self):
        # workaround for https://github.com/ray-project/ray/issues/1516
        for w in self.workers:
            w.__ray_terminate__.remote()

    def _collect_results(self):
        # get results from rollouts
        self.start = time.time()
        num_episodes, num_timesteps = 0, 0
        num_workers = self.config["num_workers"]
        results = []
        seed = np.random.randint(0,10000)
        
        rollout_ids = [{i:self.workers[(i-1) % num_workers].do_rollouts.remote(self.weights_dict[i], seed) for i in range(1, len(self.weights_dict)+1)}]

        # Get the results of the rollouts.
        res = {}
        for i in rollout_ids:
            res.update(i)
        rollout_ids = dict(sorted(res.items()))
        rollout_values = [ray.get(rollout_id) for rollout_id in rollout_ids.values()]
        for result in rollout_values:
            results.append(result)
        fitness = [x[0] for x in results]
        lengths = [x[1] for x in results]
        num_episodes = len(lengths)
        num_timesteps = sum(lengths)
        return fitness, lengths, num_episodes, num_timesteps


    def __getstate__(self):
        return {
            "algorithm_class": type(self),
            "config": self.config,
            "weights": self.weights_dict,
            "noise_std": self.noise_std
        }

    def __setstate__(self, state):
        self.weights_dict = state["weights"]
        self.config = state["config"]
        self.noise_std = state["noise_std"]


    # Resamplers
    def uni_sampler(self, fitness):
        # Stochastic Universal Resampler
        fitness = np.array(fitness)
        min_fitness = np.min(fitness)
        if min_fitness < 0:
            fitness += np.abs(min_fitness) 
        # Force more variation by making the min reward equal to 0
        if min_fitness > 0:
            fitness -= min_fitness 
        # if all rewards are the same choose a single sample and mutate the rest
        if np.all(fitness == 0):
            return {1:int(self.pop_size/2 + 1), 2: int(self.pop_size/2 - 1)}
        # multiply fitness with weights 
        fitness = np.multiply(np.array(list(self.w_all.values())), fitness)
        self.crazy = {i: fitness[i-1] for i in range(1, self.pop_size+1)}
        # Select random number from 0 to  1/N
        selected = np.random.uniform(0, 1/self.pop_size)
        # Create a list with the location of the selected particles
        points = []
        points.append(selected)
        for i in range(self.pop_size - 1):
            selected += 1/self.pop_size
            points.append(selected)
        # Normalize fitness between 0 and 1 
        zero_one_fitness = fitness/np.sum(fitness)
        norm_fitness = np.cumsum(zero_one_fitness)
        # See where the selected points fall in the fitness scale
        survivors = []
        for point in points:
            for i in range(self.pop_size):  
                if point <= norm_fitness[i]:
                    survivors.append(i)
                    break
        # Organize survivors into a dictionary with: 
        # key: sample number, and value: number of times it was selected
        survivor_dict = {}
        for survivor in survivors:
            if survivor+1 in survivor_dict:
                survivor_dict[survivor+1] += 1
            else:
                survivor_dict[survivor+1] = 1
        return survivor_dict 
    
    def res_sampler(self, fitness):
        # Residual Resampler
        min_fitness = np.min(fitness)
        if min_fitness < 0:
            fitness += np.abs(min_fitness)
        # Force more variation by making the min reward equal to 0
        if min_fitness > 0:
            fitness -= min_fitness
        # if all rewards are the same choose a single sample and mutate the rest
        if np.all(fitness == 0):
            return {1:int(self.pop_size/2 + 1), 2: int(self.pop_size/2 - 1)}
        norm_fitness = fitness/np.sum(fitness)
        z = self.pop_size * norm_fitness
        int_z = []
        survivors = []
        # Get integer values of the resampler
        for count, value in enumerate(z):
            value = int(value)
            int_z.append(value)
            if value > 0:
                for i in range(value):
                    survivors.append(count)
        #Get float values in weighted distribution
        float_z = z - int_z
        norm_float_z = np.cumsum(float_z/np.sum(float_z))
        dim = round(np.sum(float_z))
        #Randomly select points from distribution
        points = np.random.uniform(0,1, dim)
        points.sort()
        for point in points:
            for i in range(self.pop_size):  
                if point <= norm_float_z[i]:
                    survivors.append(i)
                    break
        # Organize survivors into a dictionary with: 
        # key: sample number, and value: number of times it was selected
        survivor_dict = {}
        for survivor in survivors:
            if survivor+1 in survivor_dict:
                survivor_dict[survivor+1] += 1
            else:
                survivor_dict[survivor+1] = 1
        return survivor_dict

    "add new resampler here"
    
    # Mutations
    # def mutate(self, fitness):
    #     # Mutate new population from survivors
    #     # Sort by fitness
    #     survivors = np.argsort(fitness)[round(self.pop_size*0.9)-1:]
    #     survivors += 1
    #     # Calculate weighted mean 
    #     norm_fitness = 0
    #     for i in survivors:
    #         norm_fitness += fitness[i-1]
    #         # best_sample = self.weights_dict[i]
    #     mean = 0
    #     for i in survivors:
    #         mean += (fitness[i-1]/norm_fitness) * self.weights_dict[i]    


    #     # Create new samples
    #     noise_table = np.random.normal(loc=0, scale=1, size=(len(self.weights_dict[1]), self.pop_size))
    #     noise_table *= self.noise_std
    #     # weights_table = best_sample[:,np.newaxis] + noise_table
    #     weights_table = mean[:,np.newaxis] + noise_table
    #     # create new weights dictionary
    #     self.weights_dict = {}
    #     for i, new_sample in enumerate(np.swapaxes(weights_table, 0, 1)):
    #         self.weights_dict[i+1] = new_sample
    #     # self.weights_dict[1] = best_sample

    def mutate(self, fitness):
        # Mutate new population from survivors
        # Make all fitness positive
        min_fitness = np.min(fitness)
        if min_fitness < 0:
            fitness += np.abs(min_fitness)
        if min_fitness > 0:
            fitness -= np.abs(min_fitness)
        # Sort by fitness
        survivors = np.argsort(fitness)[int(self.pop_size*0.7):]
        survivors += 1
        # Calculate weighted mean 
        norm_fitness = 0
        for i in survivors:
            norm_fitness += fitness[i-1]
            # best_sample = self.weights_dict[i]
        # Calculate mean and variance
        mean = np.zeros((10427))
        variance = np.zeros((10427))
        # mean = np.zeros((16775))
        # variance = np.zeros((16775))
        # mean = np.zeros((34055))
        # variance = np.zeros((34055))
        for i in survivors:
            mean += (fitness[i-1]/norm_fitness) * self.weights_dict[i]    
        for i in survivors:
            variance += (fitness[i-1]/norm_fitness)*((self.weights_dict[i] - self.last_mean)**2)
        noise_std = np.sqrt(variance)
        # Create new samples
        noise_table = np.random.normal(loc=0, scale=1, size=(len(self.weights_dict[1]), self.pop_size))
        noise_table = np.multiply(noise_std, np.swapaxes(noise_table, 0, 1))
        weights_table = mean + noise_table * self.noise_std
        # best = self.weights_dict[survivors[-1]]
        # weights_table = self.weights_dict[survivors[-1]] + noise_table
        # create new weights dictionary
        self.weights_dict = {}
        for i, new_sample in enumerate(weights_table):
            self.weights_dict[i+1] = new_sample
        # self.weights_dict[1] = best
        self.last_mean = mean
    
    def noise_adaptation(self, mean_fitness):
        self.noise_std *= 1

    
    def particle_filter_sampling(self, survivors):
        # Apply particle_filter Matrix Adaptation to the survivors
        # get wi from the list of survivors from Universal Resampler. wi=z/N
        self.w_all = survivors.copy()
        for key in range(1, self.pop_size+1):
            if key in survivors.keys():
                self.w_all[key] /= self.pop_size
            else: 
                self.w_all[key]= 1/self.pop_size
        # Normalize weights
        for key in self.w_all.keys():
            self.w_all[key] /= sum(list(self.w_all.values()))
        self.w_all = {k: np.array(v) for k, v in sorted(self.w_all.items(), key=lambda item: int(item[0]))}
        # Delete not selected models
        for i in range(1, self.pop_size+1):
            if i not in survivors.keys():
                del self.weights_dict[i]
                del self.crazy[i]
        crazy = np.array(list(self.crazy.values()))/sum(list(self.crazy.values()))
        for i, key in enumerate(self.crazy.keys()):
            self.crazy[key] = crazy[i]
        # Sort Dictionaries to match
        self.weights_dict = {k: np.array(v) for k, v in sorted(self.weights_dict.items(), key=lambda item: int(item[0]))}
        # Calculate Weighted mean of Gaussian
        mean_matrix = 0
        for i in survivors.keys():
            mean_matrix += self.crazy[i] * self.weights_dict[i]
        # Calculate bessel
        bessel = 1 / (1 - sum([wi ** 2 for wi in list(self.crazy.values())]))
        # Calculate Covariance Matrix
        cov_matrix = np.cov(np.array(list(self.weights_dict.values())), rowvar=False, aweights=np.array(list(self.crazy.values())))
        cov_matrix *= bessel
        # Regularization
        eigenvalue = 10**(-3)
        cov_matrix += eigenvalue*np.identity(np.array(list(self.weights_dict.values())).shape[1])
        cov_matrix = self.noise_std * cov_matrix
        self.noise_std *= 0.99 #0.05
        # LLT decomposition
        L = np.linalg.cholesky(cov_matrix)
        # Sample new models and append them to weight matrix 
        new_samples = {}
        for i in range(1, self.pop_size+1):
            if i not in self.weights_dict.keys():
                noise = np.random.normal(0, 1, size=L.shape[0])
                random_vector = (L @ noise) + mean_matrix
                new_samples[i] = np.array(random_vector)
        self.weights_dict.update(new_samples)

    # def covariance_sampling(self, fitness):
    #     # Make sure all rewards are positive
    #     min_fitness = np.min(fitness)
    #     if min_fitness < 0:
    #         fitness += np.abs(min_fitness)
        
    #     # Test with max possible reward
    #     max = 930 
    #     max_d = max - np.max(fitness)
    #     mean_d = np.max(fitness) - np.mean(fitness)
    #     N = 16775
    #     # Calculate yw
    #     survivors = np.argsort(fitness)[round(self.pop_size*0.75)-1:]
    #     survivors += 1
    #     norm_fitness = 0
    #     for i in survivors:
    #         norm_fitness += fitness[i-1]
    #     mean = 0
    #     y = []
    #     w_all = []
    #     for i in survivors:
    #         mean += (fitness[i-1]/norm_fitness) * self.weights_dict[i] 
    #         yi = (self.weights_dict[i] - self.last_mean) / self.noise_std
    #         y.append(yi)
    #         w_all.append(fitness[i-1]/norm_fitness)
    #     y = np.swapaxes(np.array(y),1,0)
    #     yw = (mean - self.last_mean)/self.noise_std
    #     # Covariance Matrix Adaptation
    #     ## New Covariance Matrix
    #     cov = y @ np.diag(np.array(w_all)) @ y.T
    #     eigenvalue = 10**(-7)
    #     cov += eigenvalue*np.identity(np.array(list(self.weights_dict.values())).shape[1])
    #     # Sampling
    #     L = np.linalg.cholesky(cov)
    #     # Noise std
    #     if mean_d > max_d*self.beta:
    #         self.noise_std *= (1-1/np.sqrt(N))
    #         self.beta *= 1.2
    #     else:
    #         self.noise_std *= (1-1/(2*np.sqrt(N)))
    #     # Sample new models and append them to weight matrix 
    #     new_samples = {}
    #     for i in range(1, self.pop_size+1):
    #         noise = np.random.normal(0, 1, size=L.shape[0])
    #         noise *= self.noise_std
    #         random_vector = (L @ noise) + mean
    #         new_samples[i] = np.array(random_vector)
    #     self.weights_dict = {}
    #     self.weights_dict.update(new_samples)
    #     # Parameter update
    #     self.last_mean = mean

    def accelerated_es(self, fitness):
        # Make sure all rewards are positive


        # Calculate yw
        survivors = np.argsort(fitness)[round(self.pop_size*0.7)-1:]
        survivors += 1 

        if np.max(fitness) > self.max:
            self.max = np.max(fitness)
            self.mean = self.weights_dict[survivors[-1]]
            self.noise_std = 0.0005
            # mean = np.zeros((10427))
            # variance = np.zeros((10427))
            mean = np.zeros((16775))
            variance = np.zeros((16775))
            for i in range(1, self.pop_size+1):
                mean +=  self.weights_dict[i] 
            mean /= self.pop_size
            for i in range(1, self.pop_size+1):        
                yi = ((self.weights_dict[i] -  self.mean)**2)
                variance += yi
            noise = np.sqrt(variance/self.pop_size)
            # self.last_cov = np.diag(noise)
            self.last_cov = np.identity(len(noise))

        fitness = [f for f in fitness if f != self.max]
        survivors = np.argsort(fitness)[round(self.pop_size*0.7)-1:]
        survivors += 1 

        # normalize weights
        min_fitness = np.min(fitness)
        if min_fitness < 0:
            fitness += np.abs(min_fitness)
        if min_fitness > 0:
            fitness -= np.abs(min_fitness)

        norm_fitness = 0
        for i in survivors:
            norm_fitness += fitness[i-1]
        y = []
        w_all = []
        for i in survivors:
            yi = (self.weights_dict[i] - self.mean) / self.noise_std
            y.append(yi)
            w_all.append(fitness[i-1]/norm_fitness)
        y = np.swapaxes(np.array(y),1,0)
        # Covariance Matrix Adaptation
        ## Evolution path
        ## New Covariance Matrix
        new_cov = y @ np.diag(np.array(w_all)) @ y.T
        cov = 0.7*self.last_cov + 0.3*new_cov
        # cov = new_cov
        eigenvalue = 10**(-8)
        cov += eigenvalue*np.identity(np.array(list(self.weights_dict.values())).shape[1])
        # Sampling
        L = np.linalg.cholesky(cov)
        # Prototype Step Size control
        # max = 930 
        # reward_mean = np.mean(fitness)
        # max_d = max - np.max(fitness)
        # mean_d = np.max(fitness) - reward_mean
        # N = 10427
        # if self.last_reward_mean > reward_mean:
        #     self.noise_std *= (1-((self.last_reward_mean)/reward_mean)/(np.sqrt(N)))
        # elif mean_d > max_d:
        #     self.noise_std *= (1-(mean_d/max_d)/(np.sqrt(N)))
        # else:
        #     self.noise_std *= (1-1/(3*np.sqrt(N)))
        # Sample new models and append them to weight matrix 
        new_samples = {}
        for i in range(1, self.pop_size+1):
            noise = np.random.normal(0, 1, size=L.shape[0])
            noise *= self.noise_std
            random_vector = (L @ noise) + self.mean
            new_samples[i] = np.array(random_vector)
        self.weights_dict = {}
        self.weights_dict.update(new_samples)
        self.weights_dict[1] = self.mean
        # Parameter update
        self.last_cov = cov
        self.noise_std *= 1.05

    def cma(self, fitness):
        # Make sure all rewards are positive
        min_fitness = np.min(fitness)
        if min_fitness < 0:
            fitness += np.abs(min_fitness)

        # Calculate yw
        survivors = np.argsort(fitness)[round(self.pop_size*0.75)-1:]
        survivors += 1
        norm_fitness = 0
        for i in survivors:
            norm_fitness += fitness[i-1]
        mean = 0
        y = []
        w_all = []
        for i in survivors:
            mean += (fitness[i-1]/norm_fitness) * self.weights_dict[i] 
            yi = (self.weights_dict[i] - self.last_mean) / self.noise_std
            y.append(yi)
            w_all.append(fitness[i-1]/norm_fitness)
        yw = np.sum(np.array([x*y for x, y in zip(y, w_all)]), axis=0)
        y = np.swapaxes(np.array(y),1,0)
        # CMA formulas from Hansen CMA Tutorial
        N = 16775
        ueff = np.sum(np.array(w_all))**2/np.sum(np.array(w_all)**2)
        cs = (ueff+2)/(N+ueff+5)
        cc = 4/(N+4) / (N+4 + 2*ueff/N)
        c1 = 2 / ((N+1.3)**2+ueff)
        cmu = np.min([1-c1, 2*(ueff-2+1/ueff) / ((N+2)**2+2*ueff/2)])
        damp = 1 + 2*np.max([0, np.sqrt((ueff-1)/(N+1))-1]) + cs
        # Step-size control
        d_2 = np.reciprocal(self.d)
        c_2 = self.low @ np.diag(d_2) @ self.low.T
        self.ps = (1-cs)*self.po + np.sqrt(cs*(2-cs)*ueff)*c_2@yw
        self.noise_std *= np.exp((cs/damp)*(np.linalg.norm(self.ps)/np.sqrt(N)-1))
        N = 16775
        # Covariance Matrix Adaptation
        ## Evolution path
        self.pc = (1-cc)*self.pc + np.sqrt(cc*(2-cc)*ueff) * yw
        ## New Covariance Matrix
        new_cov = y @ np.diag(np.array(w_all)) @ y.T
        cov = (1-c1 -cmu)*self.last_cov + c1*(self.pc @ self.pc.T) + cmu*new_cov
        # eigenvalue = 10**(-10)
        # cov += eigenvalue*np.identity(np.array(list(self.weights_dict.values())).shape[1])
        # Sampling
        self.low, d, _ = ldl(cov)
        self.d = np.sqrt(np.diagonal(d))
        L = self.low @ np.diag(self.d)
        # Sample new models and append them to weight matrix 
        new_samples = {}
        for i in range(1, self.pop_size+1):
            noise = np.random.normal(0, 1, size=L.shape[0])
            noise *= self.noise_std
            random_vector = (L @ noise) + mean
            new_samples[i] = np.array(random_vector)
        self.weights_dict = {}
        self.weights_dict.update(new_samples)
        # Parameter update
        self.last_mean = mean
        self.last_cov = cov


    
    # Method to get the mean weights of the population, calculate the reward and render
    @override(Algorithm)
    def evaluate(self, evaluation_runs = 10, render=True):
        # Turn weights_dict into weights_table
        weights_table = np.array([self.weights_dict[i] for i in range(1, self.pop_size+1)])
        mean_weights = np.mean(weights_table, axis=0)
        self.policy.set_flat_weights(mean_weights)
        rewards = {}
        len_episodes = []
        for i in range(evaluation_runs):
            obs, _ = self.env.reset()
            t = 0
            reward=0
            if render:
                while True: 
                     # act = self.policy.compute_single_action(torch.FloatTensor(obs))
                    act = self.policy.compute_single_action(obs)
                    obs, r , done, _ = self.env.step(act)
                    reward += r 
                    t +=1
                    self.env.render()
                    if done:
                        self.env.close()
                        break
            else: 
                while True: 
                     # act = self.policy.compute_single_action(torch.FloatTensor(obs))
                    act = self.policy.compute_single_action(obs)
                    obs, r , done, _ = self.env.step(act)
                    reward += r 
                    t +=1
                    if done:
                        self.env.close()
                        break
            len_episodes.append(t)
            rewards.update({i:reward})
        mean = np.array(list(rewards.values())).mean()
        print(f"Mean Rewards = {mean} \n Rewards = {rewards}, episode lengths = {len_episodes}") 

# --------------------Encoder-------------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
# Use the loaded encoder
encoder = Autoencoder().to(device)
encoder.encoder = torch.load('./model_weights/good_encoder.pth', map_location=torch.device('cpu'))
# --------------------Encoder-------------------------

# Image transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.CenterCrop(70),
    # transforms.Grayscale(1),
])


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
            nn.Linear(self.num_inputs, 10),
            nn.PReLU(),
            nn.Linear(10, 10),
            nn.PReLU(),
            nn.Linear(10, num_outputs),
        )
    # Forward Pass    
    def forward(self, input_dict, state, seq_lens):
        return self.net(input_dict["obs_flat"]), []
