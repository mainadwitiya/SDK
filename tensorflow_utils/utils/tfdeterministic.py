import os
import logging
import tensorflow as tf
import numpy as np



def settfreproduceability(seed= 42):
    """
        Function enables reproduceablity for your experiments so that you can compare easily
        Parameters:
        seed (int): seed value for global randomness
    """
    logging.warning("*** Modules are set to be deterministic , randomness in your modules will be avoided , Current seed value is {} change seed value if you want to try a different seed of parameters***".format(seed))
    #Pythonic determinism
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    #Pythonic determinism
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    # https://www.tensorflow.org/api_docs/python/tf/config/threading/set_inter_op_parallelism_threads
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    from tfdeterminism import patch
    patch()