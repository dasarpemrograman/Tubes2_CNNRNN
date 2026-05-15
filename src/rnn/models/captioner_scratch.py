import numpy as np
import tensorflow as tf
from src.rnn.layers.embedding import Embedding
from src.rnn.layers.simple_rnn import SimpleRNNLayer
from src.rnn.layers.lstm import LSTMLayer
from src.common.dense import Dense

class ImageCaptionerScratch:
    """
    Full pre-inject image captioning pipeline.
    Greedy decoding: at each step, take argmax of output distribution.
    """
    def __init__(self,
                 keras_decoder: tf.keras.Model,
                 rnn_type: str,  
                 vocab: dict,
                 idx_to_word: dict,
                 max_len: int = 40):
        self.rnn_type = rnn_type
        self.vocab = vocab
        self.idx_to_word = idx_to_word
        self.max_len = max_len

        self._load_layers(keras_decoder)

    def _load_layers(self, keras_decoder: tf.keras.Model):
        self.embedding = Embedding(keras_decoder.get_layer('embedding'))
        self.cnn_projection = Dense(keras_decoder.get_layer('cnn_projection'))

        self.rnn_layers = []
        for layer in keras_decoder.layers:
            t = type(layer).__name__
            if t == 'LSTM':
                self.rnn_layers.append(LSTMLayer(layer))
            elif t == 'SimpleRNN':
                self.rnn_layers.append(SimpleRNNLayer(layer))

        self.output_dense = Dense(keras_decoder.get_layer('output'))

    def generate_caption(self,
                         cnn_feature: np.ndarray,
                         encoder_model: tf.keras.Model = None) -> str:
        """
        cnn_feature: (2048,) — pre-extracted feature vector
        Returns: generated caption string
        """
        x_neg1 = self.cnn_projection.forward(cnn_feature)  

        if self.rnn_type == 'lstm':
            units = self.rnn_layers[0].units
            h_states = [np.zeros(self.rnn_layers[i].units, dtype=np.float32)
                        for i in range(len(self.rnn_layers))]
            c_states = [np.zeros(self.rnn_layers[i].units, dtype=np.float32)
                        for i in range(len(self.rnn_layers))]
        else:
            h_states = [np.zeros(self.rnn_layers[i].units, dtype=np.float32)
                        for i in range(len(self.rnn_layers))]

        x = x_neg1
        for i, rnn_layer in enumerate(self.rnn_layers):
            if self.rnn_type == 'lstm':
                x, h_states[i], c_states[i] = rnn_layer.forward(
                    x[np.newaxis, :],  
                    h0=h_states[i], c0=c_states[i])
                x = x[0]  
            else:
                x, h_states[i] = rnn_layer.forward(
                    x[np.newaxis, :],
                    h0=h_states[i])
                x = x[0]

        words = []
        current_token = self.vocab['<start>']

        for _ in range(self.max_len):
            emb = self.embedding.forward(np.array(current_token))

            x = emb
            for i, rnn_layer in enumerate(self.rnn_layers):
                if self.rnn_type == 'lstm':
                    x, h_states[i], c_states[i] = rnn_layer.forward(
                        x[np.newaxis, :], h0=h_states[i], c0=c_states[i])
                    x = x[0]
                else:
                    x, h_states[i] = rnn_layer.forward(
                        x[np.newaxis, :], h0=h_states[i])
                    x = x[0]

            logits = self.output_dense.forward(x) 
            current_token = int(np.argmax(logits))

            word = self.idx_to_word.get(current_token, '<unk>')
            if word == '<end>':
                break
            if word not in ('<pad>', '<start>'):
                words.append(word)

        return ' '.join(words)