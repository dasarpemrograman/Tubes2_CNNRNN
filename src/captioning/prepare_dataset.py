import os
import csv
import random

INPUT_CSV = "./data/captions.txt"

OUT_TOKEN = "./data/Flickr8k.token.txt"
OUT_TRAIN = "./data/Flickr_8k.trainImages.txt"
OUT_VAL = "./data/Flickr_8k.devImages.txt"
OUT_TEST = "./data/Flickr_8k.testImages.txt"

def prepare_data():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} tidak ditemukan. Pastikan file captions.txt sudah di dalam folder data/")
        return

    print("1. Membaca captions.txt...")
    image_captions = {}
    with open(INPUT_CSV, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) < 2: continue
            img_id = row[0].strip()
            caption = row[1].strip()
            
            if img_id not in image_captions:
                image_captions[img_id] = []
            image_captions[img_id].append(caption)

    print(f"   -> Total gambar unik: {len(image_captions)}")

    print("2. Membuat Flickr8k.token.txt (Format spesifikasi)...")
    with open(OUT_TOKEN, 'w', encoding='utf-8') as f:
        for img_id, caps in image_captions.items():
            for i, cap in enumerate(caps):
                f.write(f"{img_id}#{i}\t{cap}\n")

    print("3. Membuat file pembagian (Split) Train/Val/Test...")
    unique_images = list(image_captions.keys())

    random.seed(42) 
    random.shuffle(unique_images)

    train_imgs = unique_images[:6000]
    val_imgs = unique_images[6000:7000]
    test_imgs = unique_images[7000:]

    with open(OUT_TRAIN, 'w', encoding='utf-8') as f:
        f.write("\n".join(train_imgs) + "\n")
    with open(OUT_VAL, 'w', encoding='utf-8') as f:
        f.write("\n".join(val_imgs) + "\n")
    with open(OUT_TEST, 'w', encoding='utf-8') as f:
        f.write("\n".join(test_imgs) + "\n")

    print("Selesai! Data berhasil di-convert dan siap digunakan.")

if __name__ == "__main__":
    prepare_data()