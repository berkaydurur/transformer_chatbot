import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
# from chatbot_utils import *
# from chatbot_layers import *
import os
import time
from matplotlib import pyplot as plt
from tqdm import tqdm
import html
import random
import re
import requests

num_layers = 4
target_vocab_size = 16000
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1
storage_path = None
max_length = 40
epochs = 50
batch_size = 64
checkpoint = 5
max_checkpoint = 10
custom_checkpoint = None
num_words = 10000  # Tokenizer için oluşturulan parametreler
oov_token = '<UNK>'
pad_type = 'post'
trunc_type = 'post'
mode = "train"
eval_limit = 10
exit_phrase = ".exit"

CONVERSE_FILEPATH = "./data/movie_conversations.txt"
LINES_FILEPATH = "./data/movie_lines.txt"


def sort_data(converse_filepath, lines_filepath):
    seperator = " +++$+++ "
    url_1 = ''
    with open(converse_filepath, "r") as cf:
        cf_lines = [l for l in cf.read().split("\n") if l != ""]
        cf_fields = [f.split(seperator) for f in cf_lines]

    with open(lines_filepath, "r") as lf:
        lf_lines = [l for l in lf.read().split("\n") if l != ""]
        lf_fields = [f.split(seperator) for f in lf_lines]
        lf_dict = dict()
        for f in lf_fields:
            lf_dict[f[0]] = f[3:5]

    data = list()
    movie_batch = list()
    converse_batch = list()
    line_id1 = cf_fields[0][0]
    line_id2 = cf_fields[0][1]
    movie_id = cf_fields[0][2]

    for f in tqdm(cf_fields):
        # print(f)
        if movie_id == f[2]:

            if line_id1 == f[0] and line_id2 == f[1]:
                for idx in eval(f[3]):
                    converse_batch.append(lf_dict[idx])

            else:
                movie_batch.append(converse_batch)
                converse_batch = list()
                for idx in eval(f[3]):
                    converse_batch.append(lf_dict[idx])

            line_id1 = f[0]
            line_id2 = f[1]

        else:
            data.append(movie_batch)
            movie_batch = list()
            movie_id = f[2]

    return data


def pull_twitter(twitter_filepath, shuffle=True):
    with open(twitter_filepath, "r", encoding="utf-8") as twt_f:
        lines = twt_f.read().split("\n")

    inputs, outputs = list(), list()
    for i, l in enumerate(tqdm(lines)):
        if i % 2 == 0:
            inputs.append(bytes(html.unescape(l).lower(), "utf-8"))
        else:
            outputs.append(bytes(html.unescape(l).lower(), "utf-8"))

    popped = 0
    for i, (ins, outs) in enumerate(zip(inputs, outputs)):
        if not ins or not outs:
            ins.pop(i)
            outs.pop(i)
            popped += 1

    print(f"Pairs popped: {popped}")
    if shuffle:
        print("\nShuffling...")
        inputs, outputs = shuffle_inputs_outputs(inputs, outputs)

    return inputs, outputs


def shuffle_inputs_outputs(inputs, outputs):
    inputs_outputs = list(zip(inputs, outputs))
    random.shuffle(inputs_outputs)
    inputs, outputs = zip(*inputs_outputs)
    return inputs, outputs


def create_tokenizers(inputs_outputs, inputs_outputs_savepaths, target_vocab_size):
    inputs, outputs = inputs_outputs
    inputs_savepath, outputs_savepath = inputs_outputs_savepaths

    # create tokens using tf subword tokenizer
    inputs_tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        inputs, target_vocab_size=target_vocab_size)
    outputs_tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        outputs, target_vocab_size=target_vocab_size)
    # save tokenizers to savepaths
    print("Saving tokenizers...")
    inputs_tokenizer.save_to_file(inputs_savepath)
    outputs_tokenizer.save_to_file(outputs_savepath)

    return inputs_tokenizer, outputs_tokenizer


def load_tokenizers(inputs_outputs_savepaths):
    print("Loading tokenizers...")
    inputs_savepath, outputs_savepath = inputs_outputs_savepaths
    inputs_tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(inputs_savepath)
    outputs_tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(outputs_savepath)

    return inputs_tokenizer, outputs_tokenizer


def encode(inputs_outputs, inputs_outputs_tokenizer):
    inputs, outputs = inputs_outputs
    inputs_tokenizer, outputs_tokenizer = inputs_outputs_tokenizer

    inputs = [inputs_tokenizer.vocab_size] + inputs_tokenizer.encode(
        inputs) + [inputs_tokenizer.vocab_size + 1]

    outputs = [outputs_tokenizer.vocab_size] + outputs_tokenizer.encode(
        outputs) + [outputs_tokenizer.vocab_size + 1]

    return inputs, outputs


def tf_encode(inputs_outputs, inputs_outputs_tokenizer):
    result_in, result_out = tf.py_function(encode, [inputs_outputs, inputs_outputs_tokenizer], [tf.int64, tf.int64])
    result_in.set_shape([None])
    result_out.set_shape([None])

    return result_in, result_out


def prepare_data(batch_size, inputs_outputs, inputs_outputs_tokenizer, max_length):
    print("Preparing data...")
    inputs, outputs = inputs_outputs
    if len(inputs) == len(outputs):
        batches_in = list()
        batches_out = list()
        curr_batch_in = list()
        curr_batch_out = list()
        skipped = 0
        for (ins, outs) in zip(inputs, outputs):
            ins, outs = encode([ins, outs], inputs_outputs_tokenizer)
            if len(ins) > max_length or len(outs) > max_length:
                skipped += 1
                continue
            else:

                ins = tf.keras.preprocessing.sequence.pad_sequences(sequences=[ins], maxlen=max_length,
                                                                    padding="post", truncating='post', value=0.0)[0]
                outs = tf.keras.preprocessing.sequence.pad_sequences(sequences=[outs], maxlen=max_length,
                                                                     padding="post", truncating='post', value=0.0)[0]
                curr_batch_in.append(ins)
                curr_batch_out.append(outs)

                if len(curr_batch_in) % batch_size == 0:
                    batches_in.append(tf.convert_to_tensor(curr_batch_in, dtype=tf.int64))
                    batches_out.append(tf.convert_to_tensor(curr_batch_out, dtype=tf.int64))
                    curr_batch_in = list()
                    curr_batch_out = list()

        if curr_batch_in:
            batches_in.append(tf.convert_to_tensor(curr_batch_in, dtype=tf.int64))
            batches_out.append(tf.convert_to_tensor(curr_batch_out, dtype=tf.int64))

        print(f"Total batches per epoch: {len(batches_in)}")
        print(f"Total pairs skipped: {skipped}")

        return batches_in, batches_out

    else:
        print("Given `inputs` length is not same as `outputs` length")


def plot_attention_weights(inputs_outputs_tokenizer, attention, sentence, result, layer):
    inputs_tokenizer, outputs_tokenizer = inputs_outputs_tokenizer
    fig = plt.figure(figsize=(16, 8))

    sentence = inputs_tokenizer.encode(sentence)

    attention = tf.squeeze(attention[layer], axis=0)

    for head in range(attention.shape[0]):
        ax = fig.add_subplot(2, 4, head + 1)

        # plot the attention weights
        ax.matshow(attention[head][:-1, :], cmap='viridis')

        fontdict = {'fontsize': 10}

        ax.set_xticks(range(len(sentence) + 2))
        ax.set_yticks(range(len(result)))

        ax.set_ylim(len(result) - 1.5, -0.5)

        ax.set_xticklabels(
            ['<start>'] + [inputs_tokenizer.decode([i]) for i in sentence] + ['<end>'],
            fontdict=fontdict, rotation=90)

        ax.set_yticklabels([outputs_tokenizer.decode([i]) for i in result
                            if i < outputs_tokenizer.vocab_size],
                           fontdict=fontdict)

        ax.set_xlabel('Head {}'.format(head + 1))

    plt.tight_layout()
    plt.show()


#if __name__ == '__main__':
    # inputs, outputs = pull_twitter("./data/chat.txt")
    # print(f"Total inputs: {len(inputs)}, Total outputs: {len(outputs)}")
    # for i in range(20):
    #   print(f"""Input: {inputs[i].decode("utf-8")}""")
    #   print(f"""Output: {outputs[i].decode("utf-8")}""")
    #srt_dt = sort_data(CONVERSE_FILEPATH, LINES_FILEPATH)
    #print(srt_dt[0])


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
        # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
    return output, attention_weights


def print_out(q, k, v):
    temp_out, temp_attn = scaled_dot_product_attention(
        q, k, v, None)
    print('Attention weights are:')
    print(temp_attn)
    print('Output is:')
    print(temp_out)


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Usage:
    temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
    y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
    out, attn = temp_mha(y, k=y, q=y, mask=None)
    out.shape, attn.shape
    """

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask,
             look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def loss_function(real, pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


class TransformerChatbot(object):

    def __init__(self, config_path):

        self.num_layers = num_layers
        self.d_model = d_model
        self.dff = dff
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.max_length = max_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.target_vocab_size = target_vocab_size
        self.checkpoint = checkpoint
        self.max_checkpoint = max_checkpoint
        self.custom_checkpoint = custom_checkpoint
        self.eval_limit = eval_limit
        self.exit_phrase = exit_phrase
        self.mode = mode
        self.storage_path = "./"
        print("Degerler atanıyor...")
        self.data_path = f"{self.storage_path}data"
        self.checkpoint_path = f"{self.storage_path}checkpoints/train"
        self.tokenizer_path = f"{self.storage_path}tokenizers"
        self.inputs_savepath = f"{self.tokenizer_path}/inputs_token"
        self.outputs_savepath = f"{self.tokenizer_path}/outputs_token"


        # create folders if they don't exists to prevent errors
        if not os.path.exists(f"{self.storage_path}checkpoints"):
            os.mkdir(f"{self.storage_path}checkpoints")
        if not os.path.exists(f"{self.storage_path}checkpoints/train"):
            os.mkdir(f"{self.storage_path}checkpoints/train")
        if not os.path.exists(f"{self.storage_path}tokenizers"):
            os.mkdir(f"{self.storage_path}tokenizers")
        if not os.path.exists(f"{self.storage_path}models"):
            os.mkdir(f"{self.storage_path}models")

        # preparing tokenizers and twitter data
        self.inputs, self.outputs = pull_twitter(f"{self.data_path}/chat.txt")
        print("Twitter datası çekiliyor.")
        try:
            self.inputs_tokenizer, self.outputs_tokenizer = load_tokenizers(
                inputs_outputs_savepaths=[self.inputs_savepath, self.outputs_savepath])
            print("Tokenizer yükleniyor")
        except:
            print("No tokenizers has been created yet, creating new tokenizers...")
            self.inputs_tokenizer, self.outputs_tokenizer = create_tokenizers(
                inputs_outputs=[self.inputs, self.outputs],
                inputs_outputs_savepaths=[self.inputs_savepath, self.outputs_savepath],
                target_vocab_size=self.target_vocab_size)
            print("Yoksa tokenizer yaratılıyor.")
        self.input_vocab_size = self.inputs_tokenizer.vocab_size + 2
        self.target_vocab_size = self.outputs_tokenizer.vocab_size + 2

        self.learning_rate = CustomSchedule(self.d_model)
        print("Öğrenme oranı = {}".format(self.learning_rate))
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        print("Öğrenme loss = {}".format(self.train_loss))
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        print("Öğrenme tutarlılık = {}".format(self.train_accuracy))
        self.transformer = Transformer(
            self.num_layers, self.d_model,
            self.num_heads, self.dff,
            self.input_vocab_size,
            self.target_vocab_size,
            pe_input=self.input_vocab_size,
            pe_target=self.target_vocab_size,
            rate=self.dropout_rate)

        self.ckpt = tf.train.Checkpoint(transformer=self.transformer,
                                        optimizer=self.optimizer)
        print("Checkpoint oluşturuluyor")
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_path, max_to_keep=self.max_checkpoint)

        if self.custom_checkpoint:
            self.ckpt.restore(self.custom_checkpoint)
            print(f"Custom checkpoint restored: {self.custom_checkpoint}")
        # if a checkpoint exists, restore the latest checkpoint.
        elif self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print(f"Latest checkpoint restored: {self.ckpt_manager.latest_checkpoint}")

        if self.mode == "train":
            print("\nMODE: train\n===========\n")
            self.train_dataset = prepare_data(self.batch_size, [self.inputs, self.outputs],
                                              [self.inputs_tokenizer, self.outputs_tokenizer], self.max_length)

            self.train()
            print("Train yapılıyor")
            # do some simple evaluation after training
            for (ins, outs) in zip(self.inputs, self.outputs):
                predicted_sentence, attention_weights, sentence, result = self.translate(ins)
                print(f"\nInput: {ins}")
                print(f"Predicted: {predicted_sentence}")
                print(f"Sample output: {outs}")
            plot_attention_weights([self.inputs_tokenizer, self.outputs_tokenizer],
                                   attention_weights, sentence, result, "decoder_layer4_block2")

        elif self.mode == "eval":
            print("\nMODE: eval\n==========\n")
            self.inputs = self.inputs[:self.eval_limit]
            self.outputs = self.outputs[:self.eval_limit]

            for (ins, outs) in zip(self.inputs, self.outputs):
                predicted_sentence, attention_weights, sentence, result = self.translate(ins)
                print(f"\nInput: {ins}")
                print(f"Predicted: {predicted_sentence}")
                print(f"Sample output: {outs}")
            plot_attention_weights([self.inputs_tokenizer, self.outputs_tokenizer],
                                   attention_weights, sentence, result, "decoder_layer4_block2")

        elif self.mode == "test":
            print("\nMODE: test\n==========\n")
            while True:
                usr_input = input("[USER]: ")
                if usr_input == self.exit_phrase:
                    print("Exiting test mode...")
                    break
                else:
                    predicted_sentence, _, _, _ = self.translate(usr_input)
                    print(f"[CHABOT]: {predicted_sentence}")

    def train(self):
        # The @tf.function trace-compiles train_step into a TF graph for faster
        # execution. The function specializes to the precise shape of the argument
        # tensors. To avoid re-tracing due to the variable sequence lengths or variable
        # batch sizes (the last batch is smaller), use input_signature to specify
        # more generic shapes.

        train_step_signature = [
            tf.TensorSpec(shape=(None, None), dtype=tf.int64),
            tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        ]

        @tf.function(input_signature=train_step_signature)
        def train_step(inp, tar):
            tar_inp = tar[:, :-1]
            tar_real = tar[:, 1:]

            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

            with tf.GradientTape() as tape:
                predictions, _ = self.transformer(inp, tar_inp,
                                                  True,
                                                  enc_padding_mask,
                                                  combined_mask,
                                                  dec_padding_mask)
                loss = loss_function(tar_real, predictions)

            gradients = tape.gradient(loss, self.transformer.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))

            self.train_loss(loss)
            self.train_accuracy(tar_real, predictions)

        for epoch in range(self.epochs):
            start = time.time()

            self.train_loss.reset_states()
            self.train_accuracy.reset_states()

            # inp -> portuguese, tar -> english
            batches_in, batches_out = self.train_dataset
            for (batch, (inp, tar)) in enumerate(zip(batches_in, batches_out)):
                train_step(inp, tar)

                if batch % 100 == 0:
                    print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                        epoch + 1, batch, self.train_loss.result(), self.train_accuracy.result()))

            if (epoch + 1) % self.checkpoint == 0:
                ckpt_save_path = self.ckpt_manager.save()
                print(f"Saving checkpoint for epoch {epoch + 1} at {ckpt_save_path}")

            print("Epoch {} Loss {:.4f} Accuracy {:.4f}".format(
                epoch + 1, self.train_loss.result(), self.train_accuracy.result()))
            print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

    def evaluate(self, inp_sentence):
        start_token = [self.inputs_tokenizer.vocab_size]
        end_token = [self.inputs_tokenizer.vocab_size + 1]

        # inp sentence is portuguese, hence adding the start and end token
        inp_sentence = start_token + self.inputs_tokenizer.encode(inp_sentence) + end_token
        encoder_input = tf.expand_dims(inp_sentence, 0)

        # as the target is english, the first word to the transformer should be the
        # english start token.
        decoder_input = [self.outputs_tokenizer.vocab_size]
        output = tf.expand_dims(decoder_input, 0)

        for i in range(self.max_length):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                encoder_input, output)

            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, attention_weights = self.transformer(encoder_input,
                                                              output,
                                                              False,
                                                              enc_padding_mask,
                                                              combined_mask,
                                                              dec_padding_mask)

            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            # return the result if the predicted_id is equal to the end token
            if predicted_id == self.outputs_tokenizer.vocab_size + 1:
                return tf.squeeze(output, axis=0), attention_weights

            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=-1)

        return tf.squeeze(output, axis=0), attention_weights

    def translate(self, sentence):
        result, attention_weights = self.evaluate(sentence)

        predicted_sentence = self.outputs_tokenizer.decode([i for i in result
                                                            if i < self.outputs_tokenizer.vocab_size])

        return predicted_sentence, attention_weights, sentence, result


if __name__ == "__main__":
    srt_dt = sort_data(CONVERSE_FILEPATH, LINES_FILEPATH)
    print(srt_dt[0])
    CONFIG_PATH = "./chatbot_config.yml"
    transformer_chatbot = TransformerChatbot(CONFIG_PATH)