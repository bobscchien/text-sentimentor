import os
import tensorflow as tf

###################################################################################
################################### Hugging Face ##################################
###################################################################################

class HF2TFClassifierPredictor(tf.Module):
    def __init__(self, model_dir, pretrain_dir, preprocessors=None, max_length=256, tokenizer_params={}):
        self.model = tf.saved_model.load(model_dir)
        self.inp_lang = self.model.inp_lang.numpy().decode()        
        self.inp_bert = self.model.inp_bert.numpy().decode()        
        self.preprocessors = preprocessors
        
        self.bert_dir = os.path.join(pretrain_dir, self.inp_bert)
        self.bert_config = AutoConfig.from_pretrained(self.inp_bert, cache_dir=self.bert_dir)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.inp_bert, cache_dir=self.bert_dir, do_lower_case=True)
        self.bert_tokenizer_params = {
            'add_special_tokens':True, 
            'padding':True, 'truncation':True, 'max_length':max_length, 
            'return_attention_mask':True, 'return_token_type_ids':False
        }
        for k, v in tokenizer_params.items():
            self.bert_tokenizer_params[k] = v
        
    def __call__(self, sentence1, sentence2=None):
        # Preprocessing : bert tokenizer cannot accept byte format so we should set py_function=True
        if self.preprocessors:
            sentence1 = self.preprocessors[self.inp_lang](sentence1, py_function=True)
            if sentence2 is not None:
                sentence2 = self.preprocessors[self.inp_lang](sentence2, py_function=True)

        # Tokenization
        if sentence2 is not None:
            tokens = self.bert_tokenizer(sentence1, sentence2, return_tensors='tf', **self.bert_tokenizer_params)
        else:
            tokens = self.bert_tokenizer(sentence1, return_tensors='tf', **self.bert_tokenizer_params)            
            
        if self.bert_tokenizer_params['return_token_type_ids']:
            # Two sentences
            results = self.model(tokens['input_ids'], tokens['attention_mask'], tokens['token_type_ids'])
        else:
            # One Sentence
            results = self.model(tokens['input_ids'], tokens['attention_mask'])

        return results

###################################################################################
########################### Natural Language Processing ###########################
###################################################################################

##################################### Seq2Seq #####################################

def print_seq2seq(sentence, tokens, ground_truth=None):
    print(f'{"Input:":15s}: {sentence}')
    print(f'{"Prediction":15s}: {tokens.numpy().decode("utf-8")}')
    if ground_truth:
        print(f'{"Ground truth":15s}: {ground_truth}')
        
class Seq2SeqPredictor(tf.Module):
    def __init__(self, model, tokenizers, preprocessors, lang_pair):
        self.inp_lang, self.tar_lang = lang_pair
        self.preprocessors = preprocessors
        self.tokenizers = tokenizers
        self.model = model

    def __call__(self, sentence, max_length=20):        
        assert isinstance(sentence, tf.Tensor)
        if len(sentence.shape) == 0:
            sentence = sentence[tf.newaxis]
        
        # Preprocessing
        sentence = self.preprocessors[self.inp_lang](sentence)

        # Tokenization
        sentence = getattr(self.tokenizers, self.inp_lang).tokenize(sentence).to_tensor()
        
        encoder_input = sentence

        # Setup the first and the last
        start_end = getattr(self.tokenizers, self.tar_lang).tokenize([''])[0]
        start = start_end[0][tf.newaxis]
        end = start_end[1][tf.newaxis]

        # `tf.TensorArray` is required here (instead of a python list) so that the
        # dynamic-loop can be traced by `tf.function`.
        output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(0, start)
    
        for i in tf.range(max_length):
            output = tf.transpose(output_array.stack())
            predictions, _ = self.model([encoder_input, output], training=False)

            # select the last token from the seq_len dimension
            predictions = predictions[:, -1:]  # (batch_size, 1, vocab_size)

            # 這裡可以加入Beam Search
            predicted_id = tf.argmax(predictions, axis=-1)

            # concatentate the predicted_id to the output which is given to the TransformerDecoder
            # as its input.
            output_array = output_array.write(i+1, predicted_id[0])

            if predicted_id == end:
                break

        output = tf.transpose(output_array.stack()) # output.shape (1, tokens)
        
        # Detokenization
        text = getattr(self.tokenizers, self.tar_lang).detokenize(output)[0]
        tokens = getattr(self.tokenizers, self.tar_lang).lookup(output)[0]
        
        # Postprocessing
        text = self.preprocessors[self.tar_lang](text)
        tokens = self.preprocessors[self.tar_lang](tokens)
        
        # `tf.function` prevents us from using the attention_weights that were
        # calculated on the last iteration of the loop. So recalculate them outside
        # the loop.
        _, attention_weights = self.model([encoder_input, output[:,:-1]], training=False)

        return text, tokens, attention_weights
        
class Seq2SeqExporter(tf.Module):
    def __init__(self, predictor):
        self.predictor = predictor

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def __call__(self, sentence, max_length=100):
        (result, 
         tokens,
         attention_weights) = self.predictor(sentence, max_length=max_length)
    
        return result
        
#################################### Classifier ###################################

class ClassifierPredictor(tf.Module):
    def __init__(self, model, tokenizer, preprocessors, inp_lang, tokenizer_params={}, model_detail=''):
        self.inp_lang = inp_lang
        self.preprocessors = preprocessors        

        self.tokenizer = tokenizer
        self.tokenizer_params = tokenizer_params
        
        self.model = model
        self.model_detail = model_detail        
        
    @tf.function
    def __call__(self, sentence):
        # Preprocessing : 
        # bert tokenizer cannot accept byte format so the preoprocessed sentences should be decoded
        sentence = self.preprocessors[self.inp_lang](sentence, hugging_face=True)

        # Tokenization
        tokens = tokenizer(sentence, return_tensors='tf', **self.tokenizer_params)
        if ('return_token_type_ids' in self.tokenizer_params) & self.tokenizer_params['return_token_type_ids']:
            # Two sentences
            encoder_input = (tokens['input_ids'], tokens['attention_mask'], tokens['token_type_ids'])
        else:
            # One Sentence
            encoder_input = (tokens['input_ids'], tokens['attention_mask'])

        results = self.model(encoder_input, training=False)
        results = tf.nn.sigmoid(results)

        return results
    
###################################################################################
################################# Video Processing ################################
###################################################################################

##################################### Seq2Seq #####################################

class Video2VideoPredictor(tf.Module):
    def __init__(self, model):
        self.model = model

    def __call__(self, inputs, time_length=6):
        assert isinstance(inputs, tf.Tensor)
        # Can digest data without batch dimension axis
        if len(inputs.shape) == 4:
            inputs = inputs[tf.newaxis]
        
        # Create a empty tensor array to store the data
        output_array = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        # Append the last epoch of input data as the start token
        output_array = output_array.write(0, inputs[:, -1])
    
        for i in tf.range(time_length):
            # Rearrange the time axis
            outputs = tf.transpose(output_array.stack(), perm=[1, 0, 2, 3, 4])
            
            # Predict the results based on the latest information
            predictions, _ = self.model([inputs, outputs], training=False)

            # Select the last epoch from the seq_len dimension
            predictions = predictions[:, -1]  # (batch_size, 1, ...)
            
            # Concatentate the predictions to the outputs which is given to the TransformerDecoder
            # as its input.
            output_array = output_array.write(i, predictions)

        outputs = tf.transpose(output_array.stack(), perm=[1, 0, 2, 3, 4])
        # `tf.function` prevents us from using the attention_weights that were
        # calculated on the last iteration of the loop. So recalculate them outside
        # the loop.
        _, attention_weights = self.model([inputs, outputs], training=False)

        return outputs, attention_weights
            
class Video2VideoExporter(tf.Module):
    def __init__(self, predictor, image_shape):
        self.predictor = predictor

    #@tf.function(input_signature=[tf.TensorSpec(shape=[None, None]+list(image_shape), dtype=tf.float32)])
    def __call__(self, inputs, time_length=6):
        result, _ = self.predictor(inputs, time_length)
    
        return result
