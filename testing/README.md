# Testing the model

All testing instructions, sample data, and code are in this folder.

## Contents

- **README.md** (this file) — Instructions
- **sample_data/dem_30m_sample.tif** — 256×256 px 30m DEM (included)
- **create_sample_dem.py** — Regenerate sample: `python testing/create_sample_dem.py`
- **run_example.sh** — One-command run: `bash testing/run_example.sh` (from repo root)

## 1. Prerequisites

From repo root: clone, then `conda create -n dem-sr-v2 python=3.10`, `conda activate dem-sr-v2`, install PyTorch + numpy scipy matplotlib rasterio tqdm pyyaml tensorboard.

## 2. Checkpoint

Download **best_model.pth** from [Google Drive](https://drive.google.com/drive/folders/1ITf5GxD8aK4kbIYpleN5UEdXW8CLhSS2?usp=drive_link) → place in `outputs_v2/checkpoints/best_model.pth`.

## 3. Sample data

Included: `testing/sample_data/dem_30m_sample.tif`. To regenerate: `python testing/create_sample_dem.py` (from repo root).

## 4. Run inference

**Option A:** From repo root run `bash testing/run_example.sh`. Output: `testing/sample_data/dem_10m_sr.asc`.

**Option B — Manual:**

```bash
python inference_v2.py \
  --checkpoint outputs_v2/checkpoints/best_model.pth \
  --input testing/sample_data/dem_30m_sample.tif \
  --output testing/sample_data/dem_10m_sr.asc \
  --format aaigrid --tta --device cpu
```

Use `--device mps` or `--device cuda` for faster runs.

## 5. Verify

Open `testing/sample_data/dem_10m_sr.asc` in QGIS/ArcGIS. Resolution ~3× finer (30m → 10m).

## Quick reference

| Step | Command (from repo root) |
|------|---------------------------|
| Checkpoint | [Google Drive](https://drive.google.com/drive/folders/1ITf5GxD8aK4kbIYpleN5UEdXW8CLhSS2?usp=drive_link) → `outputs_v2/checkpoints/best_model.pth` |
| Sample | In `testing/sample_data/dem_30m_sample.tif` or `python testing/create_sample_dem.py` |
| Run | `bash testing/run_example.sh` or the manual command above |
