import importlib
import os
import sys

import numpy as np

np.random.seed({{random_seed}})
sys.path.append("../../..")

utils = importlib.import_module("utils")


model_name = "{{ model_name }}"

with open("mala.npy", "wb") as f:
    samples, rewards = utils.Sampler(
        model_name,
        dbpath=os.path.join("..", "..", "..", "posteriordb", "posterior_database"),
    ).mala()
    np.savez_compressed(f, samples=samples, rewards=rewards)
