import os
import shutil
import pathlib
import configparser

import math
import time
import tqdm
import logging
import collections
from pprint import pprint

import re
import string

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_hub as tf_hub
import tensorflow_text as tf_text
import tensorflow_datasets as tfds

config_file = '../docs/conf.ini'
config = configparser.ConfigParser()
config.read(config_file)

### Configuration

PROJECT_NAME = config['path']['PROJECT_NAME']
DIR_DATA_TOP = config['path']['DIR_DATA_TOP']

# Define Directories

DIR_MODELTENSOR = os.path.join(DIR_DATA_TOP, 'Model_Tensorflow')
DIR_MODELTORCH  = os.path.join(DIR_DATA_TOP, 'Model_Pytorch')

DIR_VOCAB      = os.path.join(DIR_DATA_TOP, 'Text_Tokenizer', 'vocab')
DIR_TOKEN      = os.path.join(DIR_DATA_TOP, 'Text_Tokenizer', 'trained')

DIR_DATA       = os.path.join(DIR_DATA_TOP, PROJECT_NAME, 'dataset')
DIR_TFRECORD   = os.path.join(DIR_DATA_TOP, PROJECT_NAME, 'datatf')
DIR_MODEL      = os.path.join(DIR_DATA_TOP, PROJECT_NAME, 'model', 'savedmodels')
DIR_CHECKPOINT = os.path.join(DIR_DATA_TOP, PROJECT_NAME, 'model', 'checkpoints')
DIR_LOG        = os.path.join(DIR_DATA_TOP, PROJECT_NAME, 'model', 'logs')
DIR_TMP        = os.path.join(DIR_DATA_TOP, PROJECT_NAME, 'tmp')

# Setup Directories

DIRs = {key:value for key, value in globals().items()}
for key, value in DIRs.items():
    if 'DIR_' in key:
        if os.path.isdir(value):
            print(f"Directory {value} exists.")
        else:
            print(f"Creating {value}...")
            os.makedirs(value)
            print(f" Succeeded!!!")
            
# Global Variables

SEED = config.getint('global', 'SEED')

### suppress warnings 

logging.getLogger('tensorflow').setLevel(logging.ERROR)  

### suppress scientific notation

np.set_printoptions(suppress=True)

### global variables

RESERVED_TOKENS = ["[PAD]", "[UNK]", "[START]", "[END]"]

##### Setup computation resources

### CPU / GPU /TPU

try: 
    # TPU Detection
    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    # Connect to the TPUs
    tf.config.experimental_connect_to_cluster(cluster_resolver)
    # This is the TPU initialization code that has to be at the beginning.
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)

    strategy = tf.distribute.TPUStrategy(cluster_resolver)
    print('\nRunning on TPU ', cluster_resolver.cluster_spec().as_dict()['worker'])
    print("All devices:", tf.config.list_logical_devices('TPU'))
except:
    physical_devices = tf.config.list_physical_devices('GPU') 
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        strategy = tf.distribute.MirroredStrategy()
        print('\nRunning on GPU...')
    else:
        strategy = tf.distribute.get_strategy()
        print('\nRunning on CPU...')
        
### RAM     

from psutil import virtual_memory

RAM_GB = virtual_memory().total / 1e9
print('\nYour runtime has {:.1f} gigabytes of available RAM\n'.format(RAM_GB))

if RAM_GB < 20:
    print('Not using a high-RAM runtime')
else:
    print('You are using a high-RAM runtime!')
