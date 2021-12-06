import re
import string
import opencc

import tqdm
import pathlib
import collections

import tensorflow as tf
import tensorflow_text as tf_text
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset

### Autotune

AUTOTUNE = tf.data.AUTOTUNE

### Chinese convertor

cc = opencc.OpenCC('s2twp')

def cc_converter(text: tf.Tensor):
    return cc.convert(text.numpy().decode())

### Global variables

RESERVED_TOKENS = ["[PAD]", "[UNK]", "[START]", "[END]"]

START = tf.argmax(tf.constant(RESERVED_TOKENS) == "[START]")
END = tf.argmax(tf.constant(RESERVED_TOKENS) == "[END]")

###################################################################################
################################ text_preprocessor ################################
###################################################################################

# Given the punctuation list
zhon_punctuation = "！＂＃＄％＆＇（）＊＋，－｡。／：；＜＝＞？＠［＼］＾＿｀｛｜｝～–….．､、《》〈〉｢｣「」『』【】〔〕‘'‛“”„‟"

# Define the function for preprocess
def zh_preprocess(text, py_function=False):
    if py_function:
        # Since the tokenizers from hugging face don't support tensor input,
        # we neet to use the native python replacement mechanism to do it.
        def preprocess(sentence):
            sentence = sentence.lower()
            sentence = re.sub("<[^>]+>", "", sentence)
            # Mandarin Phonetic Symbols are included: ㄅ-ㄦ˙ˊˇˋ
            sentence = re.sub(
                '[^\u4e00-\u9fa5\u3105-\u3129\u02CA\u02CB\u02C7\u02C9%s%s0-9]' % (re.escape(string.punctuation), zhon_punctuation), 
                '', sentence)
            # Pretrained bert tokenizers will handle the subwords so we don't add additional spaces among words
            # Replacing beginning, endding and multiple continuous spaces with a single space
            sentence = re.sub("(?<=.)(?!$)", " ", sentence)
            sentence = re.sub(r"\s\s+", " ", sentence)
            sentence = sentence.strip()
            return sentence
        
        if type(text) == str:
            text = [text]
        text = [preprocess(sentence) for sentence in text]
    else:
        text = tf.strings.lower(text)
        text = tf.strings.regex_replace(text, "<[^>]+>", "")
        # Mandarin Phonetic Symbols are included: ㄅ-ㄦ˙ˊˇˋ
        text = tf.strings.regex_replace(
            text, 
            '[^\u4e00-\u9fa5\u3105-\u3129\u02CA\u02CB\u02C7\u02C9%s%s0-9]' % (re.escape(string.punctuation), zhon_punctuation), 
            ''
        )
        # Adding a space amoung words to allow better tokenization 
        # (Therefore we don't need to replace words into spaces in the previous steps)
        text = tf.strings.regex_replace(text, "[^\s]", r" \0 ")
        # Replacing beginning, endding and multiple continuous spaces with a single space
        text = tf.strings.regex_replace(text, r"\s\s+", " ")
        text = tf.strings.strip(text)
    return text

def en_preprocess(text, py_function=False):
    if py_function:
        # Since the tokenizers from hugging face don't support tensor input,
        # we neet to use the native python replacement mechanism to do it.
        def preprocess(sentence):
            sentence = sentence.lower()
            sentence = re.sub("<[^>]+>", " ", sentence)
            sentence = re.sub('[^a-z %s%s0-9]' % (re.escape(string.punctuation), zhon_punctuation), 
                              ' ', sentence)
            # Add space between numbers to make tokenizer to only record 0-9            
            sentence = re.sub("(?<=[0-9])(?!$)", " ", sentence)
            # Replacing beginning, endding and multiple continuous spaces with a single space
            sentence = re.sub(r"\s\s+", " ", sentence)
            sentence = sentence.strip()
            return sentence
        
        if type(text) == str:
            text = [text]
        text = [preprocess(sentence) for sentence in text]
    else:
        text = tf.strings.lower(text)
        text = tf.strings.regex_replace(text, "<[^>]+>", " ")
        text = tf.strings.regex_replace(text, 
                                        '[^a-z %s%s0-9]' % (re.escape(string.punctuation), zhon_punctuation), 
                                        ' ')
        # Add space between numbers to make tokenizer to only record 0-9
        text = tf.strings.regex_replace(text, "[0-9]", r" \0 ")
        # Replacing beginning, endding and multiple continuous spaces with a single space
        text = tf.strings.regex_replace(text, r"\s\s+", " ")
        text = tf.strings.strip(text)
    return text

# Initialize the text_preprocessor for future using and saving
text_preprocessors = {
  'zhon_punctuation':zhon_punctuation,
  'zhc':zh_preprocess,
  'zh':zh_preprocess,
  'en':en_preprocess
}

###################################################################################
########################## Vocabularies Saving & Loading ##########################
###################################################################################

def write_vocab_file(filepath, vocab):
    with open(filepath, 'w') as f:
        for token in vocab:
            print(token, file=f)

def load_vocab_file(filepath):
    vocab = []
    with open(filepath, 'r') as f:
        vocab = f.read().splitlines()
    return vocab

###################################################################################
################################### Tokenization ##################################
###################################################################################

def build_bert_tokenizer(vocab_path, dataset, cjk=False,
                         bert_vocab_params={}, vocab_size=None, batch_size=1024, revocab=False):
    if (not pathlib.Path(vocab_path).is_file()) | revocab:
        # For the CJK languages, we build the vocabularies by ourselves
        # by adding spaces among words beforehand (during dataset preprocessing)
        if cjk:
            vocab = []
            vocab_dict = collections.defaultdict(lambda: 0)
            for tokens in tqdm.tqdm(dataset.prefetch(AUTOTUNE)):
                for token in tf.strings.split(tokens).numpy():
                    vocab_dict[token.decode()] += 1

            vocab = sorted(vocab_dict.items(), key=lambda x: x[1], reverse=True)
            vocab = [token for token, count in vocab]
            if vocab_size:
                vocab = vocab[:vocab_size]
            vocab = RESERVED_TOKENS + vocab
        else:
            if vocab_size:
                bert_vocab_params['vocab_size'] = vocab_size
            vocab = bert_vocab_from_dataset.bert_vocab_from_dataset(
                dataset.batch(batch_size).prefetch(AUTOTUNE),
                **bert_vocab_params
            )
        write_vocab_file(vocab_path, vocab)
    else:
        vocab = load_vocab_file(vocab_path)
    
    tokenizer = tf_text.BertTokenizer(vocab_path, 
                                      lower_case=True, 
                                      unknown_token="[UNK]")
    
    print(f'\nThere are {len(vocab)} words in the dictionary\n')
    
    return tokenizer, vocab

def demo_tokenizer(tokenizer, dataset, sample=5):
    for text in dataset.take(sample):
        # Because we don't need the extra num_tokens dimensions for our current use case
        # we can merge the last two dimensions to obtain a RaggedTensor with shape [batch, num_wordpieces]
        tokens = tokenizer.tokenize(text).merge_dims(-2,-1)
        detoken = tokenizer.detokenize(tokens).merge_dims(-2,-1)
        print('Original  :\n', text.numpy().decode())
        print('Tokenized :\n', tokens.to_list()[0])
        print('Recovered :\n', tf.strings.join(detoken, separator=' ').numpy().decode())
        print('-'*60)
        
### Postprocess

def add_start_end(ragged):
    count = ragged.bounding_shape()[0]
    starts = tf.fill([count, 1], START)
    ends = tf.fill([count, 1], END)
    return tf.concat([starts, ragged, ends], axis=1)

def cleanup_text(reserved_tokens, token_txt):
    # Drop the reserved tokens, except for "[UNK]".
    bad_tokens = [re.escape(tok) for tok in reserved_tokens if tok != "[UNK]"]
    bad_token_re = "|".join(bad_tokens)

    bad_cells = tf.strings.regex_full_match(token_txt, bad_token_re)
    result = tf.ragged.boolean_mask(token_txt, ~bad_cells)

    # Join them into strings.
    result = tf.strings.reduce_join(result, separator=' ', axis=-1)

    return result

class CustomTokenizer(tf.Module):
    def __init__(self, reserved_tokens, vocab_path):
        self.tokenizer = tf_text.BertTokenizer(vocab_path, lower_case=True, unknown_token="[UNK]")
        self._reserved_tokens = reserved_tokens
        self._vocab_path = tf.saved_model.Asset(vocab_path)

        vocab = pathlib.Path(vocab_path).read_text().splitlines()
        self.vocab = tf.Variable(vocab)

        # Create the signatures for export:   
        # Include a tokenize signature for a batch of strings. 
        self.tokenize.get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string)
        )

        # Include `detokenize` and `lookup` signatures for:
        #   * `Tensors` with shapes [tokens] and [batch, tokens]
        #   * `RaggedTensors` with shape [batch, tokens]
        self.detokenize.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int64)
        )
        self.detokenize.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64)
        )

        self.lookup.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int64)
        )
        self.lookup.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64)
        )

        # These `get_*` methods take no arguments
        self.get_vocab_size.get_concrete_function()
        self.get_vocab_path.get_concrete_function()
        self.get_reserved_tokens.get_concrete_function()

    @tf.function
    def tokenize(self, strings):
        enc = self.tokenizer.tokenize(strings)
        # Merge the `word` and `word-piece` axes.
        enc = enc.merge_dims(-2, -1)
        enc = add_start_end(enc)
        return enc

    @tf.function
    def detokenize(self, tokenized):
        words = self.tokenizer.detokenize(tokenized)
        return cleanup_text(self._reserved_tokens, words)

    @tf.function
    def lookup(self, token_ids):
        return tf.gather(self.vocab, token_ids)

    @tf.function
    def get_vocab_size(self):
        return tf.shape(self.vocab)[0]

    @tf.function
    def get_vocab_path(self):
        return self._vocab_path

    @tf.function
    def get_reserved_tokens(self):
        return tf.constant(self._reserved_tokens)
