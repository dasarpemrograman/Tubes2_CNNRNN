# Tubes2_CNNRNN

Tugas Besar 2 IF3270 Pembelajaran Mesin — Implementasi Convolutional Neural Network (CNN) dan Recurrent Neural Network (SimpleRNN & LSTM) dari scratch menggunakan NumPy, serta pelatihan model menggunakan Keras/TensorFlow.

- **CNN**: Klasifikasi citra pada dataset Intel Image Classification (6 kelas) dengan variasi hyperparameter (jumlah layer, filter, kernel, pooling) dan perbandingan shared vs non-shared parameter.
- **RNN/LSTM**: Image captioning pada dataset Flickr8k menggunakan arsitektur encoder-decoder (InceptionV3 + SimpleRNN/LSTM) dengan evaluasi BLEU-4, METEOR, analisis kualitatif, dan uji kecepatan inferensi.

## Setup

Install [uv](https://docs.astral.sh/uv/), lalu:

```powershell
uv sync
```

Requires Python 3.11 (set via `.python-version`).

## Menjalankan Program

Seluruh eksperimen dan evaluasi dilakukan melalui Jupyter Notebook:

```powershell
uv run jupyter lab
```

Buka notebook berikut:

| Notebook | Deskripsi |
| --- | --- |
| `src/notebooks/CNN_Training_Experiment_and_Evaluations.ipynb` | Training 16 konfigurasi CNN, evaluasi Keras vs scratch, shared vs non-shared parameter |
| `src/rnn/notebooks/evaluation.ipynb` | Evaluasi RNN/LSTM image captioning (kurva pembelajaran, BLEU-4/METEOR, analisis kualitatif, inference time, eksperimen max-length) |

## Penempatan Data

Dataset tidak di-commit ke git. Letakkan secara lokal:

- Intel Image Classification → `data/intel_image_classification/`
- Flickr8k → `data/` (images + annotation files)

Artifact hasil training (index, features, model weights, metrics) disimpan di `artifacts/`.

## Pembagian Tugas

| NIM | Nama | Kontribusi |
| --- | --- | --- |
| 18223009 | Muhammad Faiz Alfikrona | Pengisian laporan forward propagation LSTM dan implementasi RNN/LSTM. Setup repositori (GitHub, pyproject.toml, uv, ruff, pytest). Perancangan struktur direktori modular. Implementasi modul utilitas bersama (`src/common/`). Skrip validasi dan otomatisasi. Unit test awal. Penyusunan README dan .gitignore. |
| 18223025 | Muhammad Azzam Robbani | Implementasi from scratch dan eksperimen SimpleRNN & LSTM. Pipeline data generator (vocabulary, tokenization, ekstraksi fitur InceptionV3). Skrip pelatihan model Keras dan kelas dekoder utama (`ImageCaptionerScratch`). Optimasi layer NumPy dan stabilitas numerik. Notebook evaluasi RNN/LSTM. Penyusunan laporan (deskripsi persoalan, forward propagation, analisis hasil). |
| 18223101 | Ni Made Sekar Jelita Parameswari | Implementasi from scratch dan eksperimen CNN. Layer CNN NumPy (Conv2D, LocallyConnected2D, Pooling, Dense, dll). Pipeline preprocessing dan konfigurasi multi-eksperimen. Modul evaluasi (agregasi, Keras vs scratch, shared vs non-shared). Notebook eksperimen CNN. Dokumentasi teknis modul CNN untuk laporan. |
