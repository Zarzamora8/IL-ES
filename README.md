# IL-ES
Combination of Imitation Learning and Evolution Strategies to train Car Racing gym environment

The file accelerated_es.py uses the machine learning Pytorch and the reinforcement learning library Ray to train a model that runs the car racing environment.

The script inside ESTorchAlgo.py holds most of the logic for the creation of the parallel-running ES algorithm.

This model is initially trained using imitation learning from expert demonstrations and then a population of agents is created to start the evolution strategies process.

Three different evolution strategies algorithms are implemented including Vanilla ES, CMA-ES (covariance matrix adaptation method) and its separable version sep-CMA-ES
