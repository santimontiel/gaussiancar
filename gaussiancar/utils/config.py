import random
import numpy as np
import torch
import lightning as L
from omegaconf import OmegaConf


def register_new_resolvers():

    CUSTOM_RESOLVERS = {
        "mult": lambda x, y: x * y,
        "last_token": lambda x: x.split(".")[-1],
    }

    for name, func in CUSTOM_RESOLVERS.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, func)


def set_seed(seed: int):
    L.seed_everything(seed) 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False