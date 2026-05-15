import tensorflow as tf

def build_decoder(
    vocab_size: int,
    embed_dim: int = 256,
    rnn_units: int = 512,
    num_rnn_layers: int = 1,
    rnn_type: str = 'lstm', 
    max_caption_len: int = 40,
) -> tf.keras.Model:
    """
    Pre-inject image captioning decoder.
    Input shape: (seq_len+1, embed_dim) — CNN feature prepended
    """
    RNNClass = tf.keras.layers.LSTM if rnn_type == 'lstm' else tf.keras.layers.SimpleRNN

    caption_input = tf.keras.Input(shape=(None,), dtype=tf.int32, name='caption_input')

    cnn_input = tf.keras.Input(shape=(2048,), name='cnn_input')

    cnn_projected = tf.keras.layers.Dense(embed_dim, name='cnn_projection')(cnn_input)  

    cnn_projected = tf.keras.layers.Reshape((1, embed_dim))(cnn_projected) 

    embedded = tf.keras.layers.Embedding(vocab_size, embed_dim, name='embedding')(caption_input)

    x = tf.keras.layers.Concatenate(axis=1)([cnn_projected, embedded]) 

    for i in range(num_rnn_layers):
        return_seq = True  
        x = RNNClass(rnn_units, return_sequences=return_seq,
                     name=f'{rnn_type}_{i+1}')(x)

    outputs = tf.keras.layers.Dense(vocab_size, activation='softmax', name='output')(x)

    model = tf.keras.Model(inputs=[caption_input, cnn_input], outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model