"""
Hard-coded settings for the GUI (there is no settings dialog yet)
"""
from src.clonalg import CLONALG

# Training
EPOCHS = 15

# Test view: show "unknown" when the nearest memory cell is farther than this.
# None = never flag.
UNKNOWN_DISTANCE = None

MODEL_PARAMS = dict(
    population_size=20, # from 30
    memory_size=10, # from 5
    selection_size=3, # from 5
    num_clones=3,
    mutation_rate=0.2, # from 0.1
    similarity_threshold=7.5,
    n_neighbors=3, # from 5
    weights="distance",
    metric="euclidean",
    batch_size=16, # from None
    alpha_max=0.5,
    noise_std=0.1, # from 0.05
)


def make_model():
    """A fresh, silent model (the worker attaches its own event callback)."""
    return CLONALG(**MODEL_PARAMS, callback=None)
