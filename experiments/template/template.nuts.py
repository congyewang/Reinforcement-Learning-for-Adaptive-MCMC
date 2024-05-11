import os
import numpy as np
from utils import nuts_unconstrain_samples


model_name = "{{ model_name }}"

with open(os.path.join("baselines", model_name, "nuts.npy"), "wb") as f:
    nuts_uncon = nuts_unconstrain_samples(model_name)
    np.save(f, nuts_uncon)
