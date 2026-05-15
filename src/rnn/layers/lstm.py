import numpy as np

def sigmoid(x):
    x = np.clip(x, -500, 500) 
    return 1.0 / (1.0 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

class LSTMCell:
    def __init__(self, keras_layer=None):
        self.W_x = None   
        self.W_h = None   
        self.b   = None   
        self.units = None
        if keras_layer is not None:
            self.load_from_keras(keras_layer)

    def load_from_keras(self, keras_layer):
        weights = keras_layer.get_weights()
        self.W_x = weights[0]  
        self.W_h = weights[1]  
        self.b   = weights[2] 
        self.units = self.W_h.shape[0]

    def forward(self, x_t: np.ndarray,
                h_prev: np.ndarray,
                c_prev: np.ndarray) -> tuple:
        """
        x_t:   (input_dim,)
        h_prev: (units,)
        c_prev: (units,)
        returns: (h_t, c_t) each (units,)
        """
        u = self.units
        combined = x_t @ self.W_x + h_prev @ self.W_h + self.b
        i = sigmoid(combined[0:u])
        f = sigmoid(combined[u:2*u])
        g = tanh(combined[2*u:3*u])
        o = sigmoid(combined[3*u:4*u])
        c_t = f * c_prev + i * g
        h_t = o * tanh(c_t)
        return h_t, c_t


class LSTMLayer:
    def __init__(self, keras_layer=None):
        self.cell = LSTMCell(keras_layer)
        cfg = keras_layer.get_config() if keras_layer else {}
        self.return_sequences = cfg.get('return_sequences', False)
        self.units = self.cell.units

    def forward(self, x: np.ndarray,
                h0: np.ndarray = None,
                c0: np.ndarray = None) -> tuple:
        """
        x:  (seq_len, input_dim)
        h0, c0: (units,) or None → zeros
        returns: (output, h_final, c_final)
        """
        seq_len = x.shape[0]
        h = h0 if h0 is not None else np.zeros(self.units, dtype=np.float32)
        c = c0 if c0 is not None else np.zeros(self.units, dtype=np.float32)
        outputs = []
        for t in range(seq_len):
            h, c = self.cell.forward(x[t], h, c)
            outputs.append(h.copy())
        output = np.stack(outputs, axis=0)
        if not self.return_sequences:
            return output[-1], h, c
        return output, h, c