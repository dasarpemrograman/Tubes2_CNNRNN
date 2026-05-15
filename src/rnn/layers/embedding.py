import numpy as np

class Embedding:
    """
    Lookup table: token index → dense vector.

    Keras weight shape: (vocab_size, embed_dim)
    """
    def __init__(self, keras_layer=None):
        self.W = None 
        if keras_layer is not None:
            self.load_from_keras(keras_layer)

    def load_from_keras(self, keras_layer):
        self.W = keras_layer.get_weights()[0]

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """
        token_ids: (...) integer array
        returns: (..., embed_dim) float array
        This is just fancy indexing. No matrix multiply needed.
        """
        return self.W[token_ids]