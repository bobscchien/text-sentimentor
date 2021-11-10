import os
import shutil
import pathlib
import configparser

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

# Define Directories

DIR_MODELTENSOR = config['path']['DIR_MODELTENSOR']
DIR_MODELTORCH  = config['path']['DIR_MODELTORCH']

DIR_VOCAB = config['path']['DIR_VOCAB']
DIR_TOKEN = config['path']['DIR_TOKEN']

DIR_DATA = config['path']['DIR_DATA']
DIR_TFRECORD = config['path']['DIR_TFRECORD']
DIR_MODEL = config['path']['DIR_MODEL']
DIR_CHECKPOINT = config['path']['DIR_CHECKPOINT']
DIR_LOG = config['path']['DIR_LOG']
DIR_TMP = config['path']['DIR_TMP']

# Setup Directories

DIRs = {key:value for key, value in globals().items()}
for key, value in DIRs.items():
    if 'DIR_' in key:
        if os.path.isdir(value):
            print(f"Directory {value} exists.")
        else:
            print(f"Creating {value}...")
            os.mkdir(value)
            print(f" Succeeded!!!")
            
# Global Variables

SEED = config.getint('global', 'SEED')

### suppress warnings 

logging.getLogger('tensorflow').setLevel(logging.ERROR)  

### suppress scientific notation

np.set_printoptions(suppress=True)

### global variables

RESERVED_TOKENS = ["[PAD]", "[UNK]", "[START]", "[END]"]

### GPU

physical_devices = tf.config.list_physical_devices('GPU') 
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass