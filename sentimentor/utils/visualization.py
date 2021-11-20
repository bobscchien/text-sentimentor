import tensorflow as tf

import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

zhfont = FontProperties(fname='/usr/share/fonts/SimHei/SimHei.ttf')
mpl.rcParams['font.sans-serif']=[u'SimHei']
mpl.rcParams['axes.unicode_minus']=False

def plot_attention_head(in_tokens, translated_tokens, attention):
    # The plot is of the attention when a token was generated.
    # The model didn't generate `<START>` in the output. Skip it.
    translated_tokens = translated_tokens[1:]

    ax = plt.gca()
    ax.matshow(attention)
    ax.set_xticks(range(len(in_tokens)))
    ax.set_yticks(range(len(translated_tokens)))

    labels = [label.decode('utf-8') for label in in_tokens.numpy()]
    ax.set_xticklabels(labels, rotation=90)

    labels = [label.decode('utf-8') for label in translated_tokens.numpy()]
    ax.set_yticklabels(labels)
    
def plot_attention_weights(sentence, translated_tokens, attention_heads, tokenizers):
    
    try:
        in_tokens = tf.convert_to_tensor([sentence])
        in_tokens = tokenizers.inp.tokenize(in_tokens).to_tensor()
        in_tokens = tokenizers.inp.lookup(in_tokens)[0]
    except:
        in_tokens = tokenizers.inp.tokenize(sentence, add_special_tokens=True)
        in_tokens = tf.convert_to_tensor(in_tokens)
        translated_tokens = tf.convert_to_tensor(translated_tokens)

    heads = len(attention_heads)
    rows = math.ceil(heads/4)
    
    fig = plt.figure(figsize=(16, int(8/2)*rows))

    for h, head in enumerate(attention_heads):
        ax = fig.add_subplot(rows, 4, h+1)

        plot_attention_head(in_tokens, translated_tokens, head)

        ax.set_xlabel(f'Head {h+1}')

    plt.tight_layout()
    plt.show()