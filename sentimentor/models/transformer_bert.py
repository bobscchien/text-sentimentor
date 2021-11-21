### self-defined

from .transformer import *

### Hugging Face: https://blog.tensorflow.org/2019/11/hugging-face-state-of-art-natural.html

import datasets
import transformers
from transformers import (AutoConfig, AutoTokenizer, BertTokenizerFast, 
                          TFAutoModel, 
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
        'albert':['ckiplab/albert-tiny-chinese'],
        'roberta':['hfl/chinese-roberta-wwm-ext']
    }
}

HF_TORCH_ONLY = ['ckiplab']

def HFSelectTokenizer(bert_name):
    if 'ckiplab' in bert_name:
        return BertTokenizerFast
    else:
        return AutoTokenizer

###################################################################################
############################### tf-models-officials ###############################
###################################################################################

### Embedding Projection

# Further: bert adaptor for each attention weight
def embedding_projector(embeddings, num_projection_layers, projection_dim, activation, dropout):
    projected_embeddings = tf.keras.layers.Dense(units=projection_dim)(embeddings)
    for _ in range(num_projection_layers):
        x = tf.keras.layers.Activation(activation=activation)(projected_embeddings)
        x = tf.keras.layers.Dense(projection_dim)(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Add()([projected_embeddings, x])
        projected_embeddings = tf.keras.layers.LayerNormalization()(x)
    return projected_embeddings

class EmbeddingProjector(tf.keras.layers.Layer):
    def __init__(self, num_projection_layers, projection_dim, activation='relu', dropout=0.1):
        super(EmbeddingProjector, self).__init__()
        
        self.num_projection_layers = num_projection_layers
        
        self.init_layer = tf.keras.layers.Dense(units=projection_dim)
        self.activation_layer = tf.keras.layers.Activation(activation=activation)
        
        self.dense_layers = {}
        self.drouput_layers = {}
        self.layernorm_layers = {}
        
        # https://stackoverflow.com/questions/57517992/can-i-use-dictionary-in-keras-customized-model
        for l in range(self.num_projection_layers):
            self.dense_layers[str(l)] = tf.keras.layers.Dense(projection_dim)
            self.drouput_layers[str(l)] = tf.keras.layers.Dropout(dropout)
            self.layernorm_layers[str(l)] = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
    def call(self, embeddings, training=None):
        embeddings_projected = self.init_layer(embeddings)
        for l in range(self.num_projection_layers):
            x = self.activation_layer(embeddings_projected)
            x = self.dense_layers[str(l)](x)
            x = self.drouput_layers[str(l)](x)
            embeddings_projected = self.layernorm_layers[str(l)](x+embeddings_projected)
        return embeddings_projected
    
### TransformerEncoder

class BertTransformerEncoder(tf.keras.Model):
    def __init__(self, inp_pretrained_model, num_tune, num_projection_layers, use_lstm, nn_units, 
                 num_layers, embed_dim, num_heads, dense_dim, num_classes, activation='relu', dropout=0.1):
        super().__init__()
        
        self.num_tune = num_tune
        self.use_lstm = use_lstm
        self.nn_units = nn_units
        
        ### Load pretrained bert model

        self.inp_pretrained_model = inp_pretrained_model
        self.embedding_projector  = EmbeddingProjector(num_projection_layers, embed_dim, 
                                                       activation=activation, dropout=dropout)
        
        ### Build the downstream model
        
        self.encoder = TransformerEncoder(num_layers, embed_dim, num_heads, dense_dim, 
                                          activation=activation, dropout=dropout, embedding=False)
        
        if self.use_lstm:
            self.nn_units /= 2
            self.aggregate_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.nn_units, dropout=dropout))
        else:
            self.aggregate_layer = tf.keras.layers.GlobalMaxPool1D()
            
        self.dropout_layer = tf.keras.layers.Dropout(dropout)                
        self.dense_layer = tf.keras.layers.Dense(self.nn_units, activation=activation)
        
        if num_classes <= 2:
            num_classes = 1
        self.final_layer = tf.keras.layers.Dense(num_classes, activation=None)
        
        # Whether the fine-tune process including the pretrained model
        for layer in self.inp_pretrained_model.layers[-self.num_tune:]:
            layer.trainable = bool(self.num_tune)
            
    def call(self, inputs, training=None):
        # Keras models prefer if you pass all your inputs in the first argument
        inp, inp_mask = inputs
                    
        # Bert Embedding 
        inp_embedded = self.inp_pretrained_model(inp, attention_mask=inp_mask)[0]
        inp_embedded = self.embedding_projector(inp_embedded, training=training)

        # Encoder
        enc_outputs, _ = self.encoder(inp_embedded, mask=inp_mask, training=training)    

        # Output
        if self.use_lstm:
            x = self.aggregate_layer(enc_outputs, training=training)
        else:
            x = self.aggregate_layer(enc_outputs)
        x = self.dense_layer(x)
        x = self.dropout_layer(x, training=training)
        outputs = self.final_layer(x)

        return outputs
    
    # The most convenient method to print model.summary() 
    # similar to the sequential or functional API like.
    def build_graph(self):
        inp_ids = tf.keras.layers.Input(shape=(None, ), name='input_ids', dtype='int32')
        inp_masks = tf.keras.layers.Input(shape=(None, ), name='attention_mask', dtype='int32') 
        
        return tf.keras.Model(inputs=[inp_ids, inp_masks], 
                              outputs=self.call([inp_ids, inp_masks]))

### Transformer

class BertEncoderTransformer(tf.keras.Model):
    def __init__(self, inp_pretrained_model, num_tune, num_projection_layers, 
                 num_enc_layers, num_dec_layers, embed_dim, num_heads, dense_dim, 
                 target_vocab_size, pe_target, activation='relu', dropout=0.1, embed_pos=False):
        super().__init__()
        
        self.num_tune = num_tune

        ### Load pretrained bert model

        self.inp_pretrained_model = inp_pretrained_model
        self.embedding_projector  = EmbeddingProjector(num_projection_layers, embed_dim, 
                                                       activation=activation, dropout=dropout)
        
        ### Build the downstream model
        
        self.encoder = TransformerEncoder(num_enc_layers, embed_dim, num_heads, dense_dim, 
                                          activation=activation, dropout=dropout, embedding=False)
        self.decoder = TransformerDecoder(num_dec_layers, embed_dim, num_heads, dense_dim, 
                                          target_vocab_size=target_vocab_size, maximum_position_encoding=pe_target,
                                          activation=activation, dropout=dropout, embed_pos=embed_pos, embedding=True)        
        
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
        
        # Whether the fine-tune process including the pretrained model
        for layer in self.inp_pretrained_model.layers[-self.num_tune:]:
            layer.trainable = bool(self.num_tune)
            
    def call(self, inputs, training=None):
        # Keras models prefer if you pass all your inputs in the first argument
        [inp, inp_mask], tar = inputs
                    
        # Bert Embedding 
        inp_embedded = self.inp_pretrained_model(inp, attention_mask=inp_mask)[0]
        inp_embedded = self.embedding_projector(inp_embedded, training=training)

        # Encoder
        enc_outputs, inp_padding_mask = self.encoder(inp_embedded, mask=inp_mask, training=training)    

        # Decoder
        dec_outputs, attention_weights = self.decoder(tar, enc_outputs, inp_padding_mask, training=training)

        outputs = self.final_layer(dec_outputs)

        return outputs, attention_weights
    
    # The most convenient method to print model.summary() 
    # similar to the sequential or functional API like.
    def build_graph(self):
        inp_ids = tf.keras.layers.Input(shape=(None, ), name='input_ids', dtype='int32')
        inp_masks = tf.keras.layers.Input(shape=(None, ), name='attention_mask', dtype='int32') 
        tar_ids = tf.keras.layers.Input(shape=(None,), name='target_ids', dtype='int32')
        
        return tf.keras.Model(inputs=[[inp_ids, inp_masks], tar_ids], 
                              outputs=self.call([[inp_ids, inp_masks], tar_ids]))
