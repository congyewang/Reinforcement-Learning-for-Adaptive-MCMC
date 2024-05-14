import importlib
import os
import sys

import numpy as np

np.random.seed(0)
sys.path.append("../..")

utils = importlib.import_module("utils")


model_name = "{{ model_name }}"

with open("nuts.npy", "wb") as f:
    nuts_uncon = utils.Sampler(
        model_name, dbpath=os.path.join("..", "..", "posteriordb", "posterior_database")
    ).nuts()
    np.save(f, nuts_uncon)
