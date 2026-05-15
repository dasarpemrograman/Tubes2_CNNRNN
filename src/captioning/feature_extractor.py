import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def build_feature_extractor():
    """
    Membuat model InceptionV3 beku (frozen) untuk ekstraksi fitur.
    Output-nya berupa vektor 1D dengan ukuran (2048,) untuk setiap gambar.
    """
    print("Memuat model InceptionV3 dari Keras...")
    base_model = tf.keras.applications.InceptionV3(
        include_top=False, 
        weights='imagenet',
        pooling='avg' 
    )

    base_model.trainable = False
    return base_model

def load_and_preprocess_image(image_path):
    """
    Memuat gambar dengan PIL dan melakukan preprocessing khusus untuk InceptionV3.
    Target ukurannya adalah 299x299 piksel.
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize((299, 299))
    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.inception_v3.preprocess_input(x)
    return x

def extract_features_to_disk(image_dir: str, output_dir: str):
    """
    Melakukan iterasi ke seluruh gambar, mengekstrak fiturnya, 
    dan menyimpannya sebagai file .npy.
    """
    model = build_feature_extractor()
    image_dir_path = Path(image_dir)
    output_dir_path = Path(output_dir)

    output_dir_path.mkdir(parents=True, exist_ok=True)

    image_paths = list(image_dir_path.glob('*.jpg'))
    
    if len(image_paths) == 0:
        print(f"Tidak ada gambar ditemukan di {image_dir}. Pastikan path benar!")
        return

    print(f"Memulai ekstraksi fitur dari {len(image_paths)} gambar...")
    
    for img_path in tqdm(image_paths):
        save_path = output_dir_path / f"{img_path.stem}.npy"

        if save_path.exists():
            continue
            
        try:
            x = load_and_preprocess_image(img_path)
            feature = model.predict(x, verbose=0) 

            np.save(save_path, feature[0])
        except Exception as e:
            print(f"Gagal memproses gambar {img_path.name}: {e}")

if __name__ == "__main__":
    IMAGE_DIR = "./data/Images" 
    OUTPUT_DIR = "./data/features"
    
    extract_features_to_disk(IMAGE_DIR, OUTPUT_DIR)