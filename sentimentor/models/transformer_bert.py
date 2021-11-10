### self-defined

from .transformer import *

### Hugging Face
# https://blog.tensorflow.org/2019/11/hugging-face-state-of-art-natural.html

import transformers
import datasets

from transformers import (pipeline, AutoConfig, AutoTokenizer, TFAutoModel, 
                          TFAutoModelForSequenceClassification, 
                          TFAutoModelForTokenClassification, 
                          TFAutoModelForQuestionAnswering)
from transformers import (TFAutoModelForSeq2SeqLM)
from transformers import (TFTrainer, TFTrainingArguments,
                          AdamWeightDecay, WarmUp)



BERT_NAMES = {
    'en':{
        'bert':['bert-base-uncased'],
        'distilbert':['distilbert-base-uncased'],
        'roberta':['roberta-base'],
    },
    'zh':{
        'bert':['bert-base-chinese'],
    }
}

###################################################################################
############################### tf-models-officials ###############################
###################################################################################

### Embedding Projection

# 或許可以設計一個 adaptor
def embedding_projector(embeddings, num_projection_layers, projection_dim, activation, dropout):
    projected_embeddings = tf.keras.layers.Dense(units=projection_dim)(embeddings)
    for _ in tf.range(num_projection_layers):
        x = tf.keras.layers.Activation(activation=activation)(projected_embeddings)
        x = tf.keras.layers.Dense(projection_dim)(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Add()([projected_embeddings, x])
        projected_embeddings = tf.keras.layers.LayerNormalization()(x)
    return projected_embeddings
