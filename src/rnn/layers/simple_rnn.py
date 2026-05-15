import numpy as np

def tanh(x):
    return np.tanh(x)

class SimpleRNNCell:
    def __init__(self, keras_layer=None):
        self.W_xh = None   
        self.W_hh = None   
        self.b = None     
        if keras_layer is not None:
            self.load_from_keras(keras_layer)

    def load_from_keras(self, keras_layer):
        weights = keras_layer.get_weights()
        self.W_xh = weights[0]  
        self.W_hh = weights[1]  
        self.b    = weights[2]  

    def forward(self, x_t: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        """
        x_t:   (input_dim,)
        h_prev: (units,)
        returns: h_t (units,)
        """
        return tanh(x_t @ self.W_xh + h_prev @ self.W_hh + self.b)


class SimpleRNNLayer:
    """
    Stacks SimpleRNNCell over a sequence.
    For multi-layer: instantiate multiple SimpleRNNLayer objects.
    """
    def __init__(self, keras_layer=None):
        self.cell = SimpleRNNCell(keras_layer)
        cfg = keras_layer.get_config() if keras_layer else {}
        self.return_sequences = cfg.get('return_sequences', False)
        self.units = self.cell.W_hh.shape[0] if self.cell.W_hh is not None else None

    def forward(self, x: np.ndarray, h0: np.ndarray = None) -> tuple:
        """
        x:  (seq_len, input_dim)
        h0: (units,) or None → zeros
        returns: (output, h_final)
            output: (seq_len, units) if return_sequences else (units,)
        """
        seq_len = x.shape[0]
        h = h0 if h0 is not None else np.zeros(self.units, dtype=np.float32)
        outputs = []
        for t in range(seq_len):
            h = self.cell.forward(x[t], h)
            outputs.append(h.copy())
        output = np.stack(outputs, axis=0)  
        if not self.return_sequences:
            return output[-1], h
        return output, h