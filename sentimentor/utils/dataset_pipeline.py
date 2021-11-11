from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_datasets as tfds

### Autotune

AUTOTUNE = tf.data.AUTOTUNE

###################################################################################
############################### tensorflow tfrecord ###############################
###################################################################################

# The following functions can be used to convert a value to a type compatible with tf.train.Example.

def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

###################################################################################
############################### tensorflow datasets ###############################
###################################################################################

def setup_tfds_builder(builder, pcts, as_supervised=True):
    
    ### Set the splitting ratio

    (pct_train, pct_valid), pct_test = pcts
    pct_drop = 100-pct_train-pct_valid
    
    # Training
    
    if pct_train == 100:
        split_train = [f'train', f'train', f'train']
    else:
        split_train = [f'train[:{pct_train}%]', 
                       f'train[{pct_train}%:{pct_train+pct_valid}%]', 
                       f'train[{-pct_drop}%:]']
    
    # Testing
    
    if ('test' in builder.info.splits):
        split_name = 'test'
    elif ('validation' in builder.info.splits):
        split_name = 'validation'
    else:
        split_name = None
    
    if split_name is not None:
        split_test = [f'{split_name}[:{pct_test}%]', f'{split_name}[{-pct_test}%:]']
    else:
        split_test = None
    
    ### Create the datasets
    
    train_dataset, valid_dataset, _ = builder.as_dataset(split=split_train, as_supervised=as_supervised)
        
    if (split_test is not None) & (pct_test!=0):
        test_dataset, _ = builder.as_dataset(split=split_test, as_supervised=as_supervised)
    else:
        test_dataset = None
    
    return train_dataset, valid_dataset, test_dataset

###################################################################################
############################### tensorflow pipeline ###############################
###################################################################################

def make_batches(dataset, batch_size=64, buffer_size=None, cache=True, 
                 fn_before_cache=None, fn_before_batch=None, fn_before_prefetch=None):
        
    # Cache
    if fn_before_cache:
        dataset = dataset.map(fn_before_cache, num_parallel_calls=AUTOTUNE, deterministic=None)
    if cache:
        dataset = dataset.cache()
    
    # Shuffle
    if buffer_size:
        dataset = dataset.shuffle(buffer_size)
    
    # Batch
    if fn_before_batch:
        dataset = dataset.map(fn_before_batch, num_parallel_calls=AUTOTUNE, deterministic=None)
    dataset = dataset.batch(batch_size)
    
    # Prefetch
    if fn_before_prefetch:
        dataset = dataset.map(fn_before_prefetch, num_parallel_calls=AUTOTUNE, deterministic=None)
    dataset = dataset.prefetch(AUTOTUNE)    
    
    return dataset

def make_custom_token_pair_batches(dataset, tokenizers, max_lengths=None, 
                                   batch_size=64, buffer_size=None, cache=True):

    def tokenize_pairs(inp, tar):
        # Convert from ragged to dense, padding with zeros.
        inp = tokenizers.inp.tokenize(inp)
        tar = tokenizers.tar.tokenize(tar)
        
        # Truncate sentence
        if max_lengths:
            inp = inp[:, :max_lengths['inp']]
            tar = tar[:, :max_lengths['tar']]

        # Pad sentence
        return tf.cast(inp, dtype=tf.int32).to_tensor(), tf.cast(tar, dtype=tf.int32).to_tensor()
        #return inp.to_tensor(), tar.to_tensor()
    
    return make_batches(dataset, batch_size, buffer_size, cache,
                        fn_before_cache=None, fn_before_batch=None, fn_before_prefetch=tokenize_pairs)
