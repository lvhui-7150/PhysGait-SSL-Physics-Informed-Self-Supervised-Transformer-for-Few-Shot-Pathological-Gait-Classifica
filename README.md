# PhysGait-SSL-Physics-Informed-Self-Supervised-Transformer-for-Few-Shot-Pathological-Gait-Classifica

This repository implements a **Physics-Informed Self-Supervised Learning (SSL)** framework for gait analysis using wearable sensor data (accelerometer/gyroscope). The model leverages a masked autoencoder pretraining strategy and incorporates a physics-based loss derived from a two‑link pendulum model of the human leg, enforcing biomechanical consistency. The approach is evaluated on the **Daphnet** fall detection dataset under few‑shot settings.

## Key Features
- **Self‑supervised pretraining** with random patch masking and reconstruction loss (both masked and visible regions).
- **Physics‑informed fine‑tuning** that adds a dynamics‑based regularization term (Lagrangian mechanics of a two‑link leg model).
- **Few‑shot evaluation** (10%, 20%, 50%, 100% of training data) with multiple repetitions and subject‑wise cross‑validation.
- **Modular design** allowing easy adjustment of SSL/physics weighting, masking ratio, and network architecture.

---

## Dataset
The **Daphnet** dataset consists of accelerometer and gyroscope signals from three wearable sensors (attached to the ankle, shank, and thigh) during walking and falling activities. In this implementation:
- Input signals: 9 channels (3 axes × 3 sensors).
- Binary classification: fall vs. normal gait (labels binarised to 0/1).
- Files are expected in `.txt` format, with columns: timestamp + 9 sensor channels + label.

Place all `.txt` files under `./Daphnet`. The code automatically organises files by subject (first three characters of filename).

---

## Dependencies
- Python 3.8+
- PyTorch ≥ 1.10
- NumPy
- scikit-learn
- glob, os, math, random (standard libraries)

Install required packages:
```bash
pip install torch numpy scikit-learn
```

---

## Code Structure
- `DaphnetDataset` : loads and segments sensor data into fixed‑length windows with labels.
- `patchify / unpatchify` : convert time series into non‑overlapping patches for Transformer input.
- `PositionalEncoding` : sinusoidal positional embeddings.
- `PhysGaitSSL` : main model containing:
  - Patch embedding + Transformer encoder.
  - SSL reconstruction head.
  - Classification head.
  - Physics decoder (outputs joint angles).
- `physics_loss_fn` : computes residual dynamics loss using two‑link leg equations.
- `pretrain_stage_I` : masked autoencoding pretraining with visible region auxiliary loss.
- `finetune_stage_II` : fine‑tuning with classification + optional SSL and physics losses.
- `evaluate` + `find_best_threshold` : evaluation using Youden’s index for threshold selection.
- `main` : runs subject‑wise cross‑validation over few‑shot ratios.

---

## Configuration
All hyperparameters are defined at the top of the script. Key parameters include:
| Parameter | Description |
|-----------|-------------|
| `WINDOW` | Input window length (128 time steps) |
| `PATCH_SIZE` | Patch size (must divide `WINDOW`, e.g., 16) |
| `MASK_RATIO` | Fraction of patches to mask during pretraining (0.3) |
| `SSL_EPOCHS` / `FT_EPOCHS` | Number of pretraining and fine‑tuning epochs |
| `PHYSICS_WEIGHT` | Weight of physics loss during fine‑tuning |
| `SSL_WEIGHT_IN_FT` | Weight of SSL reconstruction loss during fine‑tuning |
| `VISIBLE_WEIGHT` | Weight for visible‑region reconstruction in pretraining |
| `FEW_SHOT_RATIOS` | List of training subset fractions to evaluate |
| `NUM_REPEATS` | Number of random subsampling repetitions per ratio |

Physical constants (mass, length, inertia, gravity) are set in `SEGMENT_PARAMS` and `GRAVITY`.

---

## Usage
Simply run the main script:
```bash
python main.py
```
The script will:
1. Load and preprocess the Daphnet dataset.
2. For each test subject, train on all other subjects’ data.
3. For each few‑shot ratio, repeat `NUM_REPEATS` times:
   - Randomly sample a subset of training data.
   - Split into train/validation (80%/20%).
   - Pretrain on **all** training data (unlabelled) using the masked autoencoder.
   - Fine‑tune on the few‑shot labelled subset (with optional physics/SSL losses).
   - Select best classification threshold on the validation set.
   - Evaluate on the held‑out test subject.
4. Print average metrics (accuracy, precision, recall, F1, AUC) across subjects and repetitions.

---

## Output Example
During execution, you will see logs for pretraining and fine‑tuning epochs. At the end, a summary like:
```
FINAL RESULTS (averaged across subjects and repeats)
Ratio 0.1: acc = 0.8234 ± 0.0213
Ratio 0.1: precision = 0.8101 ± 0.0312
...
```
Metrics are computed for each few‑shot ratio.

---

## Customisation
- **Modify network architecture** : adjust `d_model`, `nhead`, `num_layers` in `PhysGaitSSL`.
- **Change physics model** : update `SEGMENT_PARAMS` or the equations in `physics_loss_fn`.
- **Disable SSL/physics losses** : set `SSL_WEIGHT_IN_FT=0` or `PHYSICS_WEIGHT=0` during fine‑tuning.
- **Use only classification** : set both weights to zero.

---

## Notes
- The physics decoder outputs two joint angles (hip and knee) – this matches the two‑link model.
- The validation split is taken **from the few‑shot subset** to avoid data leakage.
- Deterministic behaviour is enforced via `set_seed(42)` for reproducibility.

---

---

## License
This project is provided for research purposes. Please refer to the original Daphnet dataset license for data usage.

---

## Contact
For questions or issues, please open an issue on this repository or contact the author.

---

**Enjoy exploring physics‑aware deep learning for gait recognition!**
