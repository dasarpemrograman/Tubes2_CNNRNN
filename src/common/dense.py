import numpy as np

class Dense:
    def __init__(self, keras_layer=None):
        """
        Inisialisasi Dense layer. 
        Jika keras_layer diberikan, otomatis copy bobot W, b, dan fungsi aktivasinya.
        """
        self.W = None
        self.b = None
        self.activation_name = None
        
        if keras_layer is not None:
            weights = keras_layer.get_weights()
            self.W = weights[0]
            self.b = weights[1]
            if hasattr(keras_layer, 'activation'):
                self.activation_name = keras_layer.activation.__name__

    def forward(self, x):
        out = np.dot(x, self.W) + self.b
        
        if self.activation_name == 'softmax':
            e_x = np.exp(out - np.max(out, axis=-1, keepdims=True))
            out = e_x / np.sum(e_x, axis=-1, keepdims=True)
        elif self.activation_name == 'relu':
            out = np.maximum(0, out)
            
        return out

    def __call__(self, x):
        return self.forward(x)