import os
import sys
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(script_dir))

if root_dir not in sys.path:
    sys.path.append(root_dir)

from utils import mala_unconstrain_samples


model_name = "{{ model_name }}"

with open("mala.npy", "wb") as f:
    mala_uncon = mala_unconstrain_samples(model_name, dbpath="../../posteriordb/posterior_database")
    np.save(f, mala_uncon)
