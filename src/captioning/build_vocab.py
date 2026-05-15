import os
from src.rnn.utils.caption_utils import load_flickr8k_captions, build_vocabulary, save_vocabulary

TOKEN_FILE = "./data/Flickr8k.token.txt"
TRAIN_SPLIT_FILE = "./data/Flickr_8k.trainImages.txt"
VOCAB_FILE = "./artifacts/captioning/vocabulary.json"

def load_split_ids(filepath: str) -> set:
    with open(filepath, 'r') as f:
        return set([line.strip() for line in f.readlines()])

if __name__ == "__main__":
    print("1. Load semua caption mentah...")
    all_captions = load_flickr8k_captions(TOKEN_FILE)
    
    print("2. Filter khusus data training (mencegah data leak)...")
    train_ids = load_split_ids(TRAIN_SPLIT_FILE)
    
    train_captions_list = []
    for img_id, caps in all_captions.items():
        if img_id in train_ids:
            train_captions_list.extend(caps)
            
    print(f"   -> Total kalimat latih: {len(train_captions_list)}")
    
    print("3. Membangun vocabulary (min_freq=5)...")
    vocab, _ = build_vocabulary(train_captions_list, min_freq=5)
    print(f"   -> Ukuran vocabulary final: {len(vocab)} kata")
    
    print("4. Menyimpan ke JSON...")
    os.makedirs(os.path.dirname(VOCAB_FILE), exist_ok=True)
    save_vocabulary(vocab, VOCAB_FILE)
    print(f"Selesai! Vocabulary tersimpan di: {VOCAB_FILE}")