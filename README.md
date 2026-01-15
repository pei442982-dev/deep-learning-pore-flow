# Deep Learning Pore Flow (Two-Phase)

This repository provides a minimal PyTorch baseline for predicting **two-phase flow fields** in porous media. The model predicts:

- **Velocity field** (u, v)
- **Saturation field** (Sw)

The baseline uses a UNet-style encoder/decoder that maps pore-scale input features (e.g., permeability, porosity, pressure boundary indicators) to multi-channel outputs.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train.py --data-dir /path/to/npz
```

## Expected dataset format

The training script expects `.npz` files containing:

- `inputs`: `(N, C_in, H, W)` tensor
- `targets`: `(N, 3, H, W)` tensor with channels `[u, v, Sw]`

You can adapt `src/data.py` to your data format (e.g., HDF5, TFRecords).

## Notes

- This is a baseline model intended to be extended with physics-informed losses or additional conditioning on boundary conditions and time.
- The training loop is intentionally minimal and uses MSE with channel-wise weighting.
