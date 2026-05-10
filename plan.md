Below is the **fully corrected gated to-do list**, now including **uv** as the Python environment and dependency manager.

Ownership:

| Person       | End-to-end ownership                                                                                        |
| ------------ | ----------------------------------------------------------------------------------------------------------- |
| **Person A** | CNN track                                                                                                   |
| **Person B** | RNN/LSTM image captioning track                                                                             |
| **Person C** | Everything else: repo, uv setup, tooling, Google Docs coordination, README, release, compliance, submission |

Final invariant: **the report is written in Google Docs, exported to `doc/report.pdf`, and `doc/` contains no other files.** The repository uses **uv**, so there is **no `requirements.txt`** unless the team deliberately exports one for convenience. The official dependency files are `pyproject.toml` and `uv.lock`.

This plan covers the required CNN, RNN/LSTM, from-scratch forward propagation, experiments, deliverables, and all bonus items from the spec. 

---

# Gate 0 — Scope, ownership, repo, Google Docs, and uv contract

## Person A — CNN owner

* [ ] Read the full CNN section of the spec.
* [ ] Confirm Intel Image Classification task.
* [ ] Confirm 6 class labels:

  * [ ] buildings
  * [ ] forest
  * [ ] glacier
  * [ ] mountain
  * [ ] sea
  * [ ] street
* [ ] Choose CNN input image size.
* [ ] Define CNN experiment grid with exactly 16 required architectures:

  * [ ] 2 variants of number of convolution layers.
  * [ ] 2 variants of filter combinations.
  * [ ] 2 variants of kernel-size combinations.
  * [ ] 2 variants of pooling type.
* [ ] Define CNN required artifacts:

  * [ ] Intel dataset index.
  * [ ] image preprocessing utilities.
  * [ ] CNN utility feature extractor.
  * [ ] 16 Keras model weights.
  * [ ] 16 training histories.
  * [ ] 16 prediction files.
  * [ ] macro F1 results.
  * [ ] best CNN model.
  * [ ] scratch CNN predictions.
  * [ ] Keras-vs-scratch comparison.
  * [ ] shared-vs-non-shared comparison.
  * [ ] CNN plots and tables for Google Docs.
* [ ] Define CNN bonus artifacts:

  * [ ] intermediate feature-map visualizations.
  * [ ] Grad-CAM visualizations.
  * [ ] scratch batch inference results.
  * [ ] scratch backward propagation gradient checks.
* [ ] Confirm all CNN commands will be runnable through `uv run`.
* [ ] Stop.

## Person B — RNN/LSTM owner

* [ ] Read the full RNN/LSTM image captioning section of the spec.
* [ ] Confirm Flickr8k image captioning task.
* [ ] Confirm dataset split target:

  * [ ] 6,000 train images.
  * [ ] 1,000 validation images.
  * [ ] 1,000 test images.
* [ ] Choose pretrained CNN encoder:

  * [ ] InceptionV3 or VGG16.
  * [ ] ImageNet weights.
  * [ ] classification head removed.
  * [ ] frozen encoder.
* [ ] Define caption preprocessing rules:

  * [ ] lowercase.
  * [ ] punctuation removal.
  * [ ] whitespace normalization.
  * [ ] `<start>` token.
  * [ ] `<end>` token.
  * [ ] `<pad>` token.
* [ ] Define required decoder experiment grid:

  * [ ] 3 recurrent-layer-count variants.
  * [ ] 2 hidden-size variants.
  * [ ] 6 SimpleRNN models.
  * [ ] 6 LSTM models.
  * [ ] 12 total required decoder models.
* [ ] Define RNN/LSTM required artifacts:

  * [ ] Flickr8k dataset index.
  * [ ] cleaned captions.
  * [ ] vocabulary JSON.
  * [ ] encoded captions.
  * [ ] padded captions.
  * [ ] extracted CNN image features.
  * [ ] RNN Keras weights.
  * [ ] LSTM Keras weights.
  * [ ] RNN scratch predictions.
  * [ ] LSTM scratch predictions.
  * [ ] BLEU-4 results.
  * [ ] METEOR results.
  * [ ] inference-time results.
  * [ ] qualitative examples.
  * [ ] max-caption-length results.
  * [ ] RNN/LSTM plots and tables for Google Docs.
* [ ] Define RNN/LSTM bonus artifacts:

  * [ ] init-inject model results.
  * [ ] beam search `k = 3` results.
  * [ ] beam search `k = 5` results.
  * [ ] scratch batch inference results.
  * [ ] scratch backward propagation gradient checks.
* [ ] Confirm all RNN/LSTM commands will be runnable through `uv run`.
* [ ] Stop.

## Person C — Everything else owner

* [ ] Create GitHub repository.
* [x] Create repository structure:

```text
src/
  common/
  cnn/
  captioning/
  scratch/
  experiments/
  notebooks/

doc/
  report.pdf

artifacts/
  cnn/
  captioning/
  common/
  bonus/

scripts/
README.md
pyproject.toml
uv.lock
.python-version
.gitignore
```

* [x] Ensure `doc/` contains **only**:

```text
doc/report.pdf
```

* [x] Initialize uv project:

  * [x] `uv init` or manually create `pyproject.toml`.
  * [x] choose Python version.
  * [x] create `.python-version`.
  * [x] add dependencies with `uv add`.
  * [x] add dev dependencies with `uv add --dev`.
  * [x] generate `uv.lock`.
* [x] Define standard commands:

  * [x] `uv sync`
  * [x] `uv run python ...`
  * [x] `uv run pytest`
  * [x] `uv run jupyter lab` or notebook equivalent.
* [ ] Create shared Google Docs report outside the codebase.
* [ ] Give edit access to all three members.
* [ ] Create Google Docs heading structure:

  * [ ] Cover.
  * [ ] Deskripsi Persoalan.
  * [ ] Pembahasan.
  * [ ] Penjelasan Implementasi.
  * [ ] Penjelasan Class, Attribute, Method.
  * [ ] Forward Propagation CNN.
  * [ ] Forward Propagation RNN.
  * [ ] Forward Propagation LSTM.
  * [ ] Backward Propagation Bonus.
  * [ ] Hasil Pengujian CNN.
  * [ ] Hasil Pengujian RNN/LSTM.
  * [ ] Bonus.
  * [ ] Kesimpulan dan Saran.
  * [ ] Pembagian Tugas.
  * [ ] Referensi.
  * [ ] Lampiran Form Penggunaan AI.
* [ ] Define artifact naming rules.
* [ ] Define result table format.
* [ ] Define figure naming format.
* [x] Define README structure.
* [x] Define compliance checklist:

  * [x] no plagiarism.
  * [x] no irresponsible AI use.
  * [x] no cross-group collaboration.
  * [x] group has exactly 3 people.
  * [x] group is not cross-class.
* [ ] Define submission rule:

  * [ ] C coordinates.
  * [ ] smallest NIM submits GitHub link on Edunex.
* [x] Record deadline: **15 May 2026**.
* [ ] Stop.

## Gate 0 exit criteria

* [ ] CNN scope frozen.
* [ ] RNN/LSTM scope frozen.
* [ ] Bonus scope frozen.
* [x] Repo exists.
* [x] uv project initialized.
* [x] `pyproject.toml` exists.
* [x] `uv.lock` exists.
* [x] `.python-version` exists.
* [ ] Google Docs report exists outside the repo.
* [x] `doc/` policy frozen: **only `report.pdf` goes there**.
* [x] Artifact paths agreed.
* [x] Deadline recorded.

---

# Gate 1 — Independent foundations

## Person A — CNN foundation

* [ ] Run `uv sync`.
* [ ] Verify CNN environment works with `uv run python`.
* [ ] Download Intel Image Classification dataset.
* [ ] Verify train/validation/test split.
* [ ] Verify all 6 classes.
* [ ] Build Intel dataset index CSV:

  * [ ] filepath.
  * [ ] split.
  * [ ] label name.
  * [ ] label id.
* [ ] Implement CNN image loader using **PIL/Pillow and NumPy only**:

  * [ ] open image with `PIL.Image.open`.
  * [ ] convert to RGB.
  * [ ] resize to target size.
  * [ ] convert to NumPy array.
  * [ ] normalize pixel values to `[0, 1]`.
* [ ] Implement CNN batch loader:

  * [ ] input: list of image paths.
  * [ ] output shape: `(N, H, W, C)`.
* [ ] Implement CNN utility feature extractor:

  * [ ] input: list of image paths.
  * [ ] use frozen Keras CNN encoder.
  * [ ] extract feature vectors.
  * [ ] save features to `.npy`.
  * [ ] skip recomputation if cached file exists.
* [ ] Implement Keras CNN model factory:

  * [ ] configurable convolution layer count.
  * [ ] configurable filter count.
  * [ ] configurable kernel size.
  * [ ] configurable max/average pooling.
  * [ ] Flatten or global pooling.
  * [ ] Dense classifier.
  * [ ] Sparse Categorical Crossentropy.
  * [ ] Adam optimizer.
* [ ] Implement macro F1 evaluation.
* [ ] Run tiny CNN smoke training through `uv run`.
* [ ] Save tiny model weights, history, and predictions.
* [ ] Stop before full 16-model training.

## Person B — RNN/LSTM foundation

* [ ] Run `uv sync`.
* [ ] Verify captioning environment works with `uv run python`.
* [ ] Download Flickr8k images.
* [ ] Download Flickr8k captions.
* [ ] Verify train/validation/test split.
* [ ] Verify each image has 5 captions.
* [ ] Build Flickr8k dataset index:

  * [ ] image id.
  * [ ] filepath.
  * [ ] split.
  * [ ] raw caption.
* [ ] Reuse or mirror A’s image loading utility for captioning image loading.
* [ ] Implement caption cleaning:

  * [ ] lowercase.
  * [ ] remove punctuation.
  * [ ] normalize whitespace.
  * [ ] prepend `<start>`.
  * [ ] append `<end>`.
* [ ] Build vocabulary from training captions only.
* [ ] Include special tokens:

  * [ ] `<pad>`.
  * [ ] `<start>`.
  * [ ] `<end>`.
* [ ] Save:

  * [ ] `token_to_id.json`.
  * [ ] `id_to_token.json`.
  * [ ] caption metadata.
* [ ] Encode captions.
* [ ] Pad captions.
* [ ] Implement frozen CNN encoder:

  * [ ] ImageNet weights.
  * [ ] no classification head.
  * [ ] frozen parameters.
* [ ] Extract tiny image-feature subset.
* [ ] Implement teacher forcing data format exactly:

  * [ ] input sequence: `[CNN_feature, emb(<start>), emb(S0), ..., emb(SN-1)]`.
  * [ ] target sequence: `[S0, S1, ..., S_N]`.
* [ ] Implement tiny RNN decoder smoke training.
* [ ] Implement tiny LSTM decoder smoke training.
* [ ] Use:

  * [ ] Sparse Categorical Crossentropy.
  * [ ] Adam.
* [ ] Run tiny captioning smoke scripts through `uv run`.
* [ ] Stop before full 12-model training.

## Person C — Infrastructure foundation

* [x] Finalize `pyproject.toml`.
* [x] Finalize `uv.lock`.
* [x] Add core dependencies through uv, likely including:

  * [x] numpy.
  * [x] pandas.
  * [x] pillow.
  * [x] matplotlib.
  * [x] scikit-learn.
  * [x] tensorflow / keras.
  * [x] nltk or metric dependency for BLEU/METEOR.
  * [x] tqdm.
  * [x] jupyter/ipykernel if notebooks are used.
* [x] Add dev dependencies through uv, likely including:

  * [x] pytest.
  * [x] ruff or formatter/linter if used.
* [x] Create shared path config.
* [x] Create shared seed utility.
* [x] Create logging utility.
* [x] Create JSON writer.
* [x] Create CSV writer.
* [x] Create plotting conventions.
* [x] Create artifact audit script skeleton.
* [x] Create README skeleton with uv setup:

  * [x] install uv.
  * [x] run `uv sync`.
  * [x] run scripts with `uv run`.
  * [x] run notebooks with uv-managed kernel.
* [ ] Maintain Google Docs report skeleton outside the repo.
* [x] Create compliance checklist.
* [ ] Create fresh-clone validation checklist.
* [x] Confirm no `requirements.txt`, unless intentionally exported as non-authoritative.
* [x] Confirm no `doc/report.md`, no `doc/figures/`, and no report writing in the repo.
* [ ] Stop before final report assembly.

## Gate 1 exit criteria

* [x] `uv sync` works.
* [ ] A can train tiny CNN and extract `.npy` features through `uv run`.
* [ ] B can preprocess captions and train tiny RNN/LSTM through `uv run`.
* [x] C has repo, README, uv, and Google Docs infrastructure.
* [x] Project runs from repository root.
* [x] `doc/` still contains only `report.pdf` placeholder or remains ready for final `report.pdf`.

---

# Gate 2 — Required end-to-end implementation

## Person A — CNN required implementation

* [ ] Implement full Keras CNN training pipeline.
* [ ] Implement 16 CNN config generator.
* [ ] Implement CNN from-scratch forward propagation layers:

  * [ ] Conv2D shared parameters.
  * [ ] LocallyConnected2D non-shared parameters.
  * [ ] MaxPooling2D.
  * [ ] AveragePooling2D.
  * [ ] GlobalAveragePooling2D.
  * [ ] GlobalMaxPooling2D.
  * [ ] Flatten.
  * [ ] Dense.
  * [ ] ReLU.
  * [ ] Softmax.
  * [ ] any other activation used.
* [ ] Implement Keras-to-scratch CNN weight loading:

  * [ ] Conv2D kernel shape `[kH, kW, C_in, C_out]`.
  * [ ] Conv2D bias.
  * [ ] LocallyConnected2D kernel.
  * [ ] LocallyConnected2D bias.
  * [ ] Dense weights.
  * [ ] Dense bias.
* [ ] Validate scratch Conv2D against tiny Keras model.
* [ ] Validate scratch LocallyConnected2D against tiny Keras model.
* [ ] Validate scratch pooling against tiny Keras model.
* [ ] Validate scratch Flatten against tiny Keras model.
* [ ] Validate scratch Dense against tiny Keras model.
* [ ] Ensure all CNN scripts run through `uv run`.
* [ ] Stop before full final CNN experiments.

## Person B — RNN/LSTM required implementation

* [ ] Extract full Flickr8k image features.
* [ ] Save full feature `.npy`.
* [ ] Save image-id-to-feature-index mapping.
* [ ] Implement Keras SimpleRNN decoder:

  * [ ] image feature input.
  * [ ] Dense projection to embedding dimension.
  * [ ] Embedding layer for caption tokens.
  * [ ] pre-inject concatenation.
  * [ ] SimpleRNN layer or stack.
  * [ ] Dense vocabulary output.
  * [ ] softmax.
* [ ] Implement Keras LSTM decoder:

  * [ ] same pre-inject structure.
  * [ ] LSTM layer or stack.
  * [ ] hidden state initialized to zeros.
  * [ ] cell state initialized to zeros.
* [ ] Use:

  * [ ] Sparse Categorical Crossentropy.
  * [ ] Adam.
* [ ] Implement RNN/LSTM from-scratch forward propagation layers:

  * [ ] Embedding.
  * [ ] Dense projection.
  * [ ] SimpleRNN cell.
  * [ ] LSTM cell.
  * [ ] Dense output.
  * [ ] softmax.
* [ ] Implement Keras-to-scratch RNN weight loading:

  * [ ] embedding matrix.
  * [ ] RNN kernel.
  * [ ] RNN recurrent kernel.
  * [ ] RNN bias.
  * [ ] output Dense weights.
* [ ] Implement Keras-to-scratch LSTM weight loading:

  * [ ] embedding matrix.
  * [ ] LSTM kernel.
  * [ ] LSTM recurrent kernel.
  * [ ] LSTM bias.
  * [ ] confirm Keras gate order.
  * [ ] output Dense weights.
* [ ] Validate scratch RNN against tiny Keras model.
* [ ] Validate scratch LSTM against tiny Keras model.
* [ ] Implement greedy decoding:

  * [ ] start from image feature.
  * [ ] feed projected feature at timestep `t = -1`.
  * [ ] feed `<start>`.
  * [ ] generate token by token.
  * [ ] stop at `<end>` or max length.
* [ ] Ensure all captioning scripts run through `uv run`.
* [ ] Stop before full final RNN/LSTM experiments.

## Person C — Non-domain implementation

* [x] Implement artifact audit script.
* [x] Implement report table generator for copying values into Google Docs.
* [x] Implement figure validation script for figures that will be inserted into Google Docs.
* [x] Implement README command placeholder system using uv commands.
* [ ] Prepare Google Docs formatting style:

  * [ ] heading levels.
  * [ ] table style.
  * [ ] figure caption format.
  * [ ] equation format.
  * [ ] reference format.
* [ ] Prepare final PDF export checklist.
* [x] Add uv commands to README placeholders:

  * [x] setup.
  * [x] CNN training.
  * [x] CNN evaluation.
  * [x] captioning training.
  * [x] captioning evaluation.
  * [x] scratch inference.
  * [x] bonus commands.
* [ ] Stop.

## Gate 2 exit criteria

* [ ] A has required CNN Keras and scratch code.
* [ ] B has required RNN/LSTM Keras and scratch code.
* [x] C has audit, README, uv, and Google Docs support tooling.
* [x] Smoke tests pass through `uv run`.
* [x] No report source files exist in the repo.

---

# Gate 3 — Required full experiments

## Person A — CNN required experiments

* [ ] Train all 16 CNN Keras architectures through `uv run`.
* [ ] For every CNN model, save:

  * [ ] config JSON.
  * [ ] Keras weights.
  * [ ] model summary.
  * [ ] training history.
  * [ ] validation predictions.
  * [ ] test predictions.
  * [ ] macro F1.
  * [ ] parameter count.
  * [ ] training/validation loss plot.
* [ ] Compare final prediction results for every hyperparameter variation.
* [ ] Analyze effect of:

  * [ ] number of convolution layers.
  * [ ] number of filters per layer.
  * [ ] kernel size.
  * [ ] pooling type.
* [ ] Select best shared Conv2D CNN.
* [ ] Run best shared CNN through scratch forward propagation.
* [ ] Compare Keras vs scratch:

  * [ ] macro F1.
  * [ ] final predictions.
  * [ ] mismatch examples.
  * [ ] possible numerical differences.
* [ ] Replace Conv2D with LocallyConnected2D for non-shared comparison.
* [ ] Train/evaluate non-shared model.
* [ ] Compare shared vs non-shared:

  * [ ] macro F1.
  * [ ] parameter count.
  * [ ] training loss.
  * [ ] validation loss.
  * [ ] efficiency.
  * [ ] conclusion about parameter sharing.
* [ ] Export final CNN result tables and figures to `artifacts/cnn/`.
* [ ] Stop.

## Person B — RNN/LSTM required experiments

* [ ] Train 6 SimpleRNN decoder variants through `uv run`.
* [ ] Train 6 LSTM decoder variants through `uv run`.
* [ ] For every decoder model, save:

  * [ ] config JSON.
  * [ ] Keras weights.
  * [ ] model summary.
  * [ ] training history.
  * [ ] generated test captions.
  * [ ] BLEU-4.
  * [ ] METEOR.
  * [ ] inference time.
  * [ ] training/validation loss plot.
* [ ] Analyze effect of recurrent layer count:

  * [ ] for RNN.
  * [ ] for LSTM.
* [ ] Analyze effect of hidden state size:

  * [ ] for RNN.
  * [ ] for LSTM.
* [ ] Select best RNN.
* [ ] Select best LSTM.
* [ ] Run equivalent Keras RNN and scratch RNN.
* [ ] Compare RNN Keras vs scratch:

  * [ ] BLEU-4.
  * [ ] METEOR.
  * [ ] execution time.
  * [ ] generated captions.
  * [ ] possible numerical differences.
* [ ] Run equivalent Keras LSTM and scratch LSTM.
* [ ] Compare LSTM Keras vs scratch:

  * [ ] BLEU-4.
  * [ ] METEOR.
  * [ ] execution time.
  * [ ] generated captions.
  * [ ] possible numerical differences.
* [ ] Compare RNN vs LSTM:

  * [ ] BLEU-4.
  * [ ] METEOR.
  * [ ] execution time.
  * [ ] training loss.
  * [ ] validation loss.
* [ ] Perform qualitative analysis with at least 10 images:

  * [ ] high-score examples.
  * [ ] medium-score examples.
  * [ ] low-score examples.
  * [ ] image.
  * [ ] RNN-generated caption.
  * [ ] LSTM-generated caption.
  * [ ] ground-truth captions.
  * [ ] observed failure pattern.
* [ ] Explain RNN vs LSTM differences using:

  * [ ] vanishing gradient.
  * [ ] long-term memory.
  * [ ] caption context retention.
* [ ] Run max-caption-length experiment:

  * [ ] choose best architecture among RNN/LSTM and Keras/scratch.
  * [ ] test at least 3 max-caption-length values.
  * [ ] compute BLEU-4 for each.
  * [ ] save plot.
  * [ ] write conclusion.
* [ ] Export final RNN/LSTM result tables and figures to `artifacts/captioning/`.
* [ ] Stop.

## Person C — Required experiment audit

* [x] Run artifact audit through `uv run`.
* [ ] Verify all CNN required result files exist.
* [ ] Verify all RNN/LSTM required result files exist.
* [ ] Verify all required plots exist in artifact folders.
* [ ] Verify all tables are readable.
* [ ] Verify metric names are consistent.
* [ ] Verify figures are ready for manual insertion into Google Docs.
* [ ] Verify README uv commands match actual script paths.
* [ ] Flag missing items to A or B.
* [x] Confirm `doc/` still contains only `report.pdf`.
* [ ] Stop.

## Gate 3 exit criteria

* [ ] All required CNN experiments complete.
* [ ] All required RNN/LSTM experiments complete.
* [ ] Required metrics saved.
* [ ] Required plots saved.
* [ ] Required artifacts audited.
* [ ] README uv commands are accurate.
* [x] Nothing except the final PDF is stored in `doc/`.

---

# Gate 4 — Bonus implementation

## Person A — CNN bonus work

### Bonus A1 — Intermediate feature-map visualization

* [ ] Select representative trained CNN.
* [ ] Select representative test images.
* [ ] Extract intermediate feature maps from convolutional layers.
* [ ] Visualize feature maps.
* [ ] Save figures to `artifacts/bonus/cnn/`.
* [ ] Write short interpretation in Google Docs.

### Bonus A2 — Grad-CAM

* [ ] Implement Grad-CAM for best CNN.
* [ ] Generate Grad-CAM for correct predictions.
* [ ] Generate Grad-CAM for incorrect predictions.
* [ ] Save heatmap overlays to `artifacts/bonus/cnn/`.
* [ ] Explain which regions influenced predictions in Google Docs.

### Bonus A3 — Scratch batch inference

* [ ] Modify CNN scratch forward propagation to support batch input.
* [ ] Test batch sizes:

  * [ ] 1.
  * [ ] 8.
  * [ ] 16 or 32.
* [ ] Compare batch output with repeated single-image output.
* [ ] Measure speed difference.
* [ ] Save results to `artifacts/bonus/cnn/`.

### Bonus A4 — CNN backward propagation from scratch

* [ ] Implement backward pass for CNN layers used:

  * [ ] Conv2D.
  * [ ] LocallyConnected2D.
  * [ ] MaxPooling2D.
  * [ ] AveragePooling2D.
  * [ ] GlobalAveragePooling2D.
  * [ ] GlobalMaxPooling2D.
  * [ ] Flatten.
  * [ ] Dense.
  * [ ] ReLU.
  * [ ] Softmax plus cross-entropy gradient.
* [ ] Validate gradients with numerical gradient checking on tiny inputs.
* [ ] Save gradient-check results to `artifacts/bonus/cnn/`.
* [ ] Optionally run tiny from-scratch CNN training loop.
* [ ] Ensure all CNN bonus commands run through `uv run`.
* [ ] Stop.

## Person B — RNN/LSTM bonus work

### Bonus B1 — Init-inject architecture

* [ ] Implement init-inject image captioning architecture.
* [ ] Use image feature to initialize recurrent state or merge image feature with caption context after recurrent processing.
* [ ] Train init-inject RNN.
* [ ] Train init-inject LSTM.
* [ ] Compare pre-inject vs init-inject:

  * [ ] BLEU-4.
  * [ ] METEOR.
  * [ ] inference time.
  * [ ] qualitative captions.
* [ ] Save results to `artifacts/bonus/captioning/`.

### Bonus B2 — Beam search

* [ ] Implement beam search decoder with `k = 3`.
* [ ] Implement beam search decoder with `k = 5`.
* [ ] Run beam search for best RNN.
* [ ] Run beam search for best LSTM.
* [ ] Compare greedy vs beam search:

  * [ ] BLEU-4.
  * [ ] METEOR.
  * [ ] inference time.
  * [ ] qualitative examples.
* [ ] Save results to `artifacts/bonus/captioning/`.

### Bonus B3 — Scratch batch inference

* [ ] Modify scratch RNN inference to support batch input.
* [ ] Modify scratch LSTM inference to support batch input.
* [ ] Test batch sizes:

  * [ ] 1.
  * [ ] 8.
  * [ ] 16 or 32.
* [ ] Compare batch output with repeated single-example output.
* [ ] Measure speed difference.
* [ ] Save results to `artifacts/bonus/captioning/`.

### Bonus B4 — RNN/LSTM backward propagation from scratch

* [ ] Implement backward pass for:

  * [ ] Embedding.
  * [ ] Dense projection.
  * [ ] SimpleRNN cell.
  * [ ] LSTM cell.
  * [ ] Dense output.
  * [ ] softmax plus cross-entropy gradient.
* [ ] Implement backpropagation through time for SimpleRNN.
* [ ] Implement backpropagation through time for LSTM.
* [ ] Validate gradients with numerical gradient checking on tiny sequences.
* [ ] Save gradient-check results to `artifacts/bonus/captioning/`.
* [ ] Optionally run tiny from-scratch RNN/LSTM training loop.
* [ ] Ensure all RNN/LSTM bonus commands run through `uv run`.
* [ ] Stop.

## Person C — Bonus integration audit

* [ ] Verify CNN bonus artifacts exist:

  * [ ] feature maps.
  * [ ] Grad-CAM.
  * [ ] batch inference.
  * [ ] backward propagation gradient checks.
* [ ] Verify RNN/LSTM bonus artifacts exist:

  * [ ] init-inject.
  * [ ] beam search.
  * [ ] batch inference.
  * [ ] backward propagation gradient checks.
* [x] Verify bonus commands are documented with `uv run` in README.
* [ ] Make sure bonus work is clearly marked as bonus in Google Docs.
* [x] Confirm no bonus report files are saved in `doc/`.
* [ ] Stop.

## Gate 4 exit criteria

* [ ] All bonus work complete.
* [ ] Bonus metrics saved.
* [ ] Bonus figures saved.
* [ ] Bonus commands run through `uv run`.
* [ ] Bonus sections ready in Google Docs.
* [x] `doc/` still only contains `report.pdf`.

---

# Gate 5 — Google Docs analysis writing

## Person A — CNN Google Docs section

* [ ] Write CNN dataset description in Google Docs.
* [ ] Write CNN preprocessing explanation in Google Docs.
* [ ] Write CNN utility feature extractor explanation in Google Docs.
* [ ] Write Keras CNN architecture explanation in Google Docs.
* [ ] Write class/attribute/method explanation for CNN code in Google Docs.
* [ ] Write CNN forward propagation explanation:

  * [ ] Conv2D.
  * [ ] LocallyConnected2D.
  * [ ] pooling.
  * [ ] global pooling.
  * [ ] Flatten.
  * [ ] Dense.
  * [ ] activations.
* [ ] Write CNN backward propagation bonus explanation.
* [ ] Insert CNN result tables into Google Docs.
* [ ] Insert CNN figures into Google Docs.
* [ ] Add captions to CNN figures.
* [ ] Write required CNN analysis:

  * [ ] effect of convolution layer count.
  * [ ] effect of filter count.
  * [ ] effect of kernel size.
  * [ ] effect of pooling type.
  * [ ] Keras vs scratch.
  * [ ] shared vs non-shared.
* [ ] Write CNN bonus analysis:

  * [ ] feature-map visualization.
  * [ ] Grad-CAM.
  * [ ] batch inference.
  * [ ] backward propagation.
* [ ] Write CNN conclusion.
* [ ] Mention all CNN commands are run with uv where relevant.
* [ ] Mark CNN section ready for C review.
* [ ] Stop.

## Person B — RNN/LSTM Google Docs section

* [ ] Write Flickr8k dataset description in Google Docs.
* [ ] Write caption preprocessing explanation in Google Docs.
* [ ] Write vocabulary explanation in Google Docs.
* [ ] Write frozen CNN encoder feature extraction explanation in Google Docs.
* [ ] Write teacher forcing explanation with exact shifted target format.
* [ ] Write pre-inject architecture explanation.
* [ ] Write Keras RNN architecture explanation.
* [ ] Write Keras LSTM architecture explanation.
* [ ] Write class/attribute/method explanation for RNN/LSTM code.
* [ ] Write RNN forward propagation explanation.
* [ ] Write LSTM forward propagation explanation.
* [ ] Write RNN/LSTM backward propagation bonus explanation.
* [ ] Write greedy decoding explanation.
* [ ] Insert RNN/LSTM result tables.
* [ ] Insert RNN/LSTM figures.
* [ ] Insert at least 10 qualitative image examples.
* [ ] Add captions to all figures.
* [ ] Write required RNN/LSTM analysis:

  * [ ] layer count.
  * [ ] hidden state size.
  * [ ] RNN vs LSTM.
  * [ ] Keras vs scratch.
  * [ ] max caption length.
  * [ ] qualitative analysis.
* [ ] Explain RNN vs LSTM using:

  * [ ] vanishing gradient.
  * [ ] long-term memory.
  * [ ] caption context retention.
* [ ] Write bonus analysis:

  * [ ] init-inject.
  * [ ] beam search.
  * [ ] batch inference.
  * [ ] backward propagation.
* [ ] Write RNN/LSTM conclusion.
* [ ] Mention all RNN/LSTM commands are run with uv where relevant.
* [ ] Mark RNN/LSTM section ready for C review.
* [ ] Stop.

## Person C — Google Docs assembly

* [ ] Maintain Google Docs formatting.
* [ ] Normalize heading levels.
* [ ] Normalize table style.
* [ ] Normalize figure captions.
* [ ] Add cover.
* [ ] Add general problem description.
* [ ] Add general discussion.
* [ ] Add task division section.
* [ ] Add references.
* [ ] Add AI usage form.
* [ ] Add compliance statement.
* [ ] Add environment/setup note:

  * [ ] project uses uv.
  * [ ] dependencies are defined in `pyproject.toml`.
  * [ ] reproducible lockfile is `uv.lock`.
* [ ] Check all required report sections exist:

  * [ ] cover.
  * [ ] problem description.
  * [ ] discussion.
  * [ ] implementation explanation.
  * [ ] class/attribute/method explanation.
  * [ ] CNN forward propagation.
  * [ ] RNN forward propagation.
  * [ ] LSTM forward propagation.
  * [ ] CNN results.
  * [ ] RNN/LSTM results.
  * [ ] conclusion and suggestions.
  * [ ] task division.
  * [ ] references.
  * [ ] AI usage form.
  * [ ] bonus section.
* [ ] Do **not** create any report source file in the repo.
* [ ] Stop before final PDF export.

## Gate 5 exit criteria

* [ ] A’s CNN section complete in Google Docs.
* [ ] B’s RNN/LSTM section complete in Google Docs.
* [ ] C’s non-domain sections complete in Google Docs.
* [ ] Full Google Docs report draft exists outside the repo.
* [ ] Google Docs mentions uv setup where appropriate.
* [ ] No report writing exists in the codebase.

---

# Gate 6 — Release candidate

## Person A — CNN release check

* [ ] Verify CNN source code exists in `src/`.
* [ ] Verify CNN testing notebooks exist in `src/notebooks/` or equivalent.
* [ ] Verify CNN commands in README use `uv run`.
* [ ] Verify CNN commands actually work after `uv sync`.
* [ ] Verify CNN metrics in Google Docs match artifact files.
* [ ] Verify CNN figures render correctly in Google Docs.
* [ ] Verify CNN bonus figures render correctly.
* [ ] Approve CNN content.
* [ ] Stop.

## Person B — RNN/LSTM release check

* [ ] Verify RNN/LSTM source code exists in `src/`.
* [ ] Verify RNN/LSTM testing notebooks exist in `src/notebooks/` or equivalent.
* [ ] Verify RNN/LSTM commands in README use `uv run`.
* [ ] Verify RNN/LSTM commands actually work after `uv sync`.
* [ ] Verify RNN/LSTM metrics in Google Docs match artifact files.
* [ ] Verify qualitative examples render correctly in Google Docs.
* [ ] Verify RNN/LSTM bonus figures render correctly.
* [ ] Approve RNN/LSTM content.
* [ ] Stop.

## Person C — Final release assembly

* [x] Finalize `README.md`:

  * [x] repo description.
  * [x] uv installation note.
  * [x] `uv sync` setup command.
  * [x] dataset placement.
  * [x] training commands using `uv run`.
  * [x] evaluation commands using `uv run`.
  * [x] scratch inference commands using `uv run`.
  * [x] bonus commands using `uv run`.
  * [x] notebook instructions using uv environment/kernel.
  * [x] task division.
  * [x] note that report was authored in Google Docs and exported to PDF.
* [x] Finalize `pyproject.toml`.
* [x] Finalize `uv.lock`.
* [x] Finalize `.python-version`.
* [x] Remove or avoid `requirements.txt` unless clearly marked as optional export.
* [x] Finalize `.gitignore`.
* [ ] Export Google Docs report as PDF.
* [ ] Save exported PDF as:

```text
doc/report.pdf
```

* [x] Verify `doc/` contains exactly one file:

```text
report.pdf
```

* [ ] Verify exported PDF includes:

  * [ ] all text.
  * [ ] all tables.
  * [ ] all figures.
  * [ ] all equations.
  * [ ] all captions.
  * [ ] references.
  * [ ] AI usage form.
* [ ] Verify `src/` contains source code and testing notebooks.
* [x] Remove local absolute paths.
* [x] Remove unnecessary large files.
* [ ] Verify GitHub repository visibility.
* [ ] Stop.

## Gate 6 exit criteria

* [ ] A approves CNN.
* [ ] B approves RNN/LSTM.
* [ ] C approves repo, README, uv setup, and exported PDF.
* [x] `pyproject.toml` exists.
* [x] `uv.lock` exists.
* [x] `.python-version` exists.
* [x] `doc/report.pdf` exists.
* [x] `doc/` contains no other files.
* [ ] Release candidate ready.

---

# Gate 7 — Fresh-clone validation and submission

## Person A — CNN fresh-clone validation

* [ ] Fresh clone repository.
* [ ] Run `uv sync`.
* [ ] Run CNN required inference command with `uv run`.
* [ ] Run CNN evaluation command with `uv run`.
* [ ] Run CNN bonus commands with `uv run`:

  * [ ] feature-map visualization.
  * [ ] Grad-CAM.
  * [ ] batch inference.
  * [ ] gradient check.
* [ ] Confirm CNN results match `doc/report.pdf`.
* [ ] Report blockers only.
* [ ] Stop.

## Person B — RNN/LSTM fresh-clone validation

* [ ] Fresh clone repository.
* [ ] Run `uv sync`.
* [ ] Run RNN/LSTM required inference command with `uv run`.
* [ ] Run RNN/LSTM evaluation command with `uv run`.
* [ ] Run RNN/LSTM bonus commands with `uv run`:

  * [ ] init-inject.
  * [ ] beam search.
  * [ ] batch inference.
  * [ ] gradient check.
* [ ] Confirm RNN/LSTM results match `doc/report.pdf`.
* [ ] Report blockers only.
* [ ] Stop.

## Person C — Final submission coordination

* [ ] Fresh clone repository.
* [x] Run `uv sync`.
* [x] Verify basic commands work with `uv run`.
* [ ] Open `doc/report.pdf`.
* [x] Check `doc/` contains exactly one file:

  * [x] `report.pdf`.
* [x] Check `src/` exists.
* [x] Check testing notebooks exist.
* [x] Check `README.md` exists.
* [x] Check `pyproject.toml` exists.
* [x] Check `uv.lock` exists.
* [x] Check `.python-version` exists.
* [ ] Check AI usage form is included inside `report.pdf`.
* [ ] Check task division is included inside `report.pdf`.
* [ ] Check references are included inside `report.pdf`.
* [x] Check compliance checklist:

  * [x] no plagiarism.
  * [x] responsible AI use.
  * [x] no cross-group collaboration.
  * [x] 3-person group.
  * [x] no cross-class group.
* [ ] Check GitHub link works in private/incognito browser.
* [ ] Confirm smallest NIM member submits GitHub link to Edunex.
* [ ] Confirm submission before **15 May 2026**.
* [ ] Save submission screenshot outside the repo.
* [ ] Notify team.
* [ ] Stop.

## Gate 7 exit criteria

* [ ] Fresh clone works.
* [x] `uv sync` works.
* [ ] Required work works through `uv run`.
* [ ] Bonus work works through `uv run`.
* [ ] Google Docs report exported to `doc/report.pdf`.
* [x] `doc/` contains only `report.pdf`.
* [ ] GitHub link accessible.
* [ ] Edunex submission complete.
* [ ] Submission screenshot saved outside the repo.

---

# Final dependency graph

```text
Gate 0 — scope, ownership, repo, Google Docs, uv contract
  ↓
Gate 1 — independent foundations
  ↓
Gate 2 — required end-to-end implementation
  ↓
Gate 3 — required full experiments
  ↓
Gate 4 — all bonus implementation
  ↓
Gate 5 — Google Docs analysis writing
  ↓
Gate 6 — export Google Docs to doc/report.pdf and finalize uv repo
  ↓
Gate 7 — fresh-clone uv validation and submission
```

Final invariants:

```text
doc/
  report.pdf
```

```text
pyproject.toml
uv.lock
.python-version
```

The report lives in Google Docs until final export. The codebase contains only the exported PDF in `doc/`, and all runnable project commands use **uv**.
