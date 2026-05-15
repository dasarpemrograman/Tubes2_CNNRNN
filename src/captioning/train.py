import json
import os
import tensorflow as tf
from src.rnn.utils.caption_utils import load_flickr8k_captions, load_vocabulary
from src.captioning.data_generator import CaptionDataGenerator
from src.rnn.models.decoder_keras import build_decoder

FEATURE_DIR = "./data/features"
TOKEN_FILE = "./data/Flickr8k.token.txt"
TRAIN_SPLIT_FILE = "./data/Flickr_8k.trainImages.txt"
VAL_SPLIT_FILE = "./data/Flickr_8k.devImages.txt"

VOCAB_FILE = "./artifacts/captioning/vocabulary.json"
WEIGHTS_DIR = "./artifacts/captioning/weights"

def load_split_ids(filepath: str) -> set:
    """Membaca daftar nama file gambar untuk split train/val."""
    with open(filepath, 'r') as f:
        return set([line.strip() for line in f.readlines()])

def run_experiment(exp_id, rnn_type, num_layers, rnn_units, train_gen, val_gen, vocab_size):
    print(f"\n{'='*50}")
    print(f"MULAI TRAINING: {exp_id}")
    print(f"Type: {rnn_type.upper()} | Layers: {num_layers} | Units: {rnn_units}")
    print(f"{'='*50}")
    
    model = build_decoder(
        vocab_size=vocab_size,
        embed_dim=256,
        rnn_units=rnn_units,
        num_rnn_layers=num_layers,
        rnn_type=rnn_type
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,                   
        restore_best_weights=True,    
        verbose=1
    )

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=20,
        callbacks=[early_stopping],
        verbose=1
    )

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    weight_path = os.path.join(WEIGHTS_DIR, f"{exp_id}.weights.h5")
    model.save_weights(weight_path)

    history_path = os.path.join(WEIGHTS_DIR, f"{exp_id}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history.history, f, indent=2)
        
    print(f"[SELESAI] Model dan history tersimpan untuk: {exp_id}\n")


if __name__ == "__main__":
    print("Memuat Vocabulary dan Data Caption...")
    vocab, _ = load_vocabulary(VOCAB_FILE)
    vocab_size = len(vocab)
    all_captions = load_flickr8k_captions(TOKEN_FILE)

    train_ids = load_split_ids(TRAIN_SPLIT_FILE)
    val_ids = load_split_ids(VAL_SPLIT_FILE)
    
    train_dict = {img_id: caps for img_id, caps in all_captions.items() if img_id in train_ids}
    val_dict = {img_id: caps for img_id, caps in all_captions.items() if img_id in val_ids}
    
    print(f"Jumlah Data Latih: {len(train_dict)} gambar")
    print(f"Jumlah Data Validasi: {len(val_dict)} gambar")

    max_len = 40
    batch_size = 64
    
    print("Membuat Data Generator...")
    train_gen = CaptionDataGenerator(train_dict, FEATURE_DIR, vocab, max_len, batch_size)
    val_gen = CaptionDataGenerator(val_dict, FEATURE_DIR, vocab, max_len, batch_size, shuffle=False)

    experiments = [
        {"id": "rnn-L1-U128", "type": "rnn",  "layers": 1, "units": 128},
        {"id": "rnn-L1-U512", "type": "rnn",  "layers": 1, "units": 512},
        {"id": "rnn-L2-U128", "type": "rnn",  "layers": 2, "units": 128},
        {"id": "rnn-L2-U512", "type": "rnn",  "layers": 2, "units": 512},
        {"id": "rnn-L3-U128", "type": "rnn",  "layers": 3, "units": 128},
        {"id": "rnn-L3-U512", "type": "rnn",  "layers": 3, "units": 512},
        
        {"id": "lstm-L1-U128", "type": "lstm", "layers": 1, "units": 128},
        {"id": "lstm-L1-U512", "type": "lstm", "layers": 1, "units": 512},
        {"id": "lstm-L2-U128", "type": "lstm", "layers": 2, "units": 128},
        {"id": "lstm-L2-U512", "type": "lstm", "layers": 2, "units": 512},
        {"id": "lstm-L3-U128", "type": "lstm", "layers": 3, "units": 128},
        {"id": "lstm-L3-U512", "type": "lstm", "layers": 3, "units": 512},
    ]

    for exp in experiments:
        if os.path.exists(os.path.join(WEIGHTS_DIR, f"{exp['id']}.weights.h5")):
            print(f"Melewati {exp['id']} (sudah di-training sebelumnya).")
            continue
            
        run_experiment(
            exp_id=exp["id"],
            rnn_type=exp["type"],
            num_layers=exp["layers"],
            rnn_units=exp["units"],
            train_gen=train_gen,
            val_gen=val_gen,
            vocab_size=vocab_size
        )