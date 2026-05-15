import re
import json
import numpy as np
from collections import Counter
from pathlib import Path

SPECIAL_TOKENS = {'<pad>': 0, '<start>': 1, '<end>': 2, '<unk>': 3}

def clean_caption(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

def build_vocabulary(captions: list[str],
                     min_freq: int = 5) -> tuple[dict, dict]:
    """
    Build word→idx and idx→word mappings.
    min_freq=5 is the standard for Flickr8k. Don't go lower.
    """
    counter = Counter()
    for cap in captions:
        counter.update(cap.split())

    vocab = dict(SPECIAL_TOKENS)
    idx = len(vocab)
    for word, freq in sorted(counter.items()):
        if freq >= min_freq and word not in vocab:
            vocab[word] = idx
            idx += 1

    idx_to_word = {v: k for k, v in vocab.items()}
    return vocab, idx_to_word

def encode_caption(caption: str, vocab: dict, max_len: int) -> np.ndarray:
    """
    Encode caption to integer sequence WITH <start> and <end>, padded.
    Returns: (max_len+2,) array
    """
    tokens = ['<start>'] + caption.split() + ['<end>']
    ids = [vocab.get(t, vocab['<unk>']) for t in tokens]
    ids = ids[:max_len+2]
    ids += [vocab['<pad>']] * (max_len + 2 - len(ids))
    return np.array(ids, dtype=np.int32)

def load_flickr8k_captions(token_file: str) -> dict[str, list[str]]:
    """
    Parse Flickr8k.token.txt
    Returns: {image_id: [caption1, ..., caption5]}
    """
    captions = {}
    with open(token_file) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            img_id = parts[0].split('#')[0]
            caption = clean_caption(parts[1])
            captions.setdefault(img_id, []).append(caption)
    return captions

def save_vocabulary(vocab: dict, path: str):
    with open(path, 'w') as f:
        json.dump(vocab, f, indent=2)

def load_vocabulary(path: str) -> tuple[dict, dict]:
    with open(path) as f:
        vocab = json.load(f)
    idx_to_word = {int(v): k for k, v in vocab.items()}
    return vocab, idx_to_word