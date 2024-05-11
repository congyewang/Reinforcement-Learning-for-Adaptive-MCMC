import os
import sys
import numpy as np

np.random.seed(0)


script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(script_dir))

if root_dir not in sys.path:
    sys.path.append(root_dir)

from utils import nuts_unconstrain_samples


model_name = "{{ model_name }}"

with open("nuts.npy", "wb") as f:
    nuts_uncon = nuts_unconstrain_samples(
        model_name, dbpath="../../posteriordb/posterior_database"
    )
    np.save(f, nuts_uncon)
