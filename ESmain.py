import ray
from ESTorchAlgo import ESConfig, ES, EvolutionNN, MyCallbacks
from ray.rllib.models import ModelCatalog

ray.init(address='local', num_cpus=10)
# Get config
config = ESConfig()
# Register Model
ModelCatalog.register_custom_model("torch_model", EvolutionNN)
# Create a Algorithm
algo = ES(env="CarRacing-v2", config={
    "env_config" : {"continuous": False,
                    'lap_complete_percent': 1,
                    # 'render_mode':"human"
                    },
    "framework": "torch",
    "episodes_per_batch": 1,
    "pop_size": 50,
    "noise_std": 1,
    "mutation": "normal",
    "sampler": "universal",
    "num_rollout_workers": 10,
    "num_gpus": 0,
    "model": {  
    "custom_model": "torch_model",
    "custom_model_config": {},
    },
})
# Train
if __name__ == "__main__":
    for i in range(100):
        algo.train()
    path_to_checkpoint = algo.save()
    print(
    "An Algorithm checkpoint has been created inside directory: "
    f"'{path_to_checkpoint}'."
    )