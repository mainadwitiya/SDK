import os
import torch
import numpy as np
import logging
import random

def setpytorchreproduceability(seed = 42):
    """
        Function enables reproduceablity for your experiments so that you can compare easily
        Parameters:
        seed (int): seed value for global randomness
    """

    logging.warning("*** Modules are set to be deterministic , randomness in your modules will be avoided , Current seed value is {} change seed value if you want to try a different seed of parameters***".format(seed))
    #Pythonic determinism
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    #Pythonic determinism
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True