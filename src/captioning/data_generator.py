import os
import numpy as np
import tensorflow as tf

class CaptionDataGenerator(tf.keras.utils.Sequence):
    """
    Menghasilkan batch data secara dinamis untuk model pre-inject Keras.
    Mencegah RAM jebol saat training karena data di-load per batch.
    """
    def __init__(self, captions_dict, feature_dir, vocab, max_len, batch_size=32, shuffle=True):
        self.data = []
        for img_id, caps in captions_dict.items():
            for cap in caps:
                self.data.append((img_id, cap))
                
        self.feature_dir = feature_dir
        self.vocab = vocab
        self.max_len = max_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.indexes = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        batch_idxs = self.indexes[index*self.batch_size : (index+1)*self.batch_size]
        
        cnn_inputs = []
        caption_inputs = []
        targets = []
        
        for idx in batch_idxs:
            img_id, caption_text = self.data[idx]

            feature_path = os.path.join(self.feature_dir, f"{img_id}.npy")
            if os.path.exists(feature_path):
                feature = np.load(feature_path)
            else:
                feature = np.zeros((2048,), dtype=np.float32)

            tokens = ['<start>'] + caption_text.split() + ['<end>']
            token_ids = [self.vocab.get(t, self.vocab.get('<unk>', 3)) for t in tokens]

            full_seq = token_ids[:self.max_len + 2]
            full_seq += [self.vocab.get('<pad>', 0)] * ((self.max_len + 2) - len(full_seq))

            text_input = full_seq[:-1]
            target_seq = full_seq[:] 
            
            cnn_inputs.append(feature)
            caption_inputs.append(text_input)
            targets.append(target_seq)
            
        return {
            'caption_input': np.array(caption_inputs, dtype=np.int32),
            'cnn_input': np.array(cnn_inputs, dtype=np.float32)
        }, np.array(targets, dtype=np.int32)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)