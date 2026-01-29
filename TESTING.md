# Testing the Model

Short instructions to run inference and test 30m→10m DEM super-resolution with sample data or your own 30m DEM.

---

## 1. Prerequisites

- **Python 3.10+** with conda/pip
- **Checkpoint:** `best_model.pth` (~181 MB) — see [Getting the checkpoint](#2-getting-the-checkpoint) below.

```bash
git clone https://github.com/Allexzaa/RCAN-DEM-v2.git
cd RCAN-DEM-v2

conda create -n dem-sr-v2 python=3.10
conda activate dem-sr-v2
pip install torch torchvision torchaudio
pip install numpy scipy matplotlib rasterio tqdm pyyaml tensorboard
```

---

## 2. Getting the checkpoint

`best_model.pth` is too large for the repo (>100 MB). Use one of:

- **Google Drive (recommended):** Download **best_model.pth** from [this folder](https://drive.google.com/drive/folders/1ITf5GxD8aK4kbIYpleN5UEdXW8CLhSS2?usp=drive_link). Place it in `outputs_v2/checkpoints/best_model.pth`.
- **GitHub Releases:** Check [Releases](https://github.com/Allexzaa/RCAN-DEM-v2/releases) for a `best_model.pth` asset. Download it and place it in `outputs_v2/checkpoints/best_model.pth`.
- **Git LFS:** If the repo uses LFS for that file, run `git lfs pull` after cloning.
- **Your own:** If you trained the model, use your saved `best_model.pth` in `outputs_v2/checkpoints/`.

---

## 3. Sample data (quick test)

A small 30m DEM can be generated so you can run inference without external data.

### Option A: Create the sample (recommended)

```bash
python scripts/create_sample_dem.py
```

This creates `sample_data/dem_30m_sample.tif` (256×256 px, synthetic terrain).

### Option B: Download a real 30m tile (optional)

With [AWS CLI](https://aws.amazon.com/cli/) and [GDAL](https://gdal.org/):

```bash
mkdir -p sample_data
aws s3 cp s3://copernicus-dem-30m/Copernicus_DSM_COG_10_N39_00_W106_00_DEM/Copernicus_DSM_COG_10_N39_00_W106_00_DEM.tif sample_data/ --no-sign-request --quiet
gdal_translate -projwin -106 39.25 -105.75 39 sample_data/Copernicus_DSM_COG_10_N39_00_W106_00_DEM.tif sample_data/dem_30m_colorado.tif
# Use sample_data/dem_30m_colorado.tif as --input
```

---

## 4. Run inference

**Using the script-generated sample:**

```bash
python inference_v2.py \
  --checkpoint outputs_v2/checkpoints/best_model.pth \
  --input sample_data/dem_30m_sample.tif \
  --output sample_data/dem_10m_sr.asc \
  --format aaigrid \
  --device cpu
```

Use `--device mps` on Apple Silicon or `--device cuda` on NVIDIA GPU.

**Output:** `sample_data/dem_10m_sr.asc` — 10m super-resolved DEM in AAIGrid format (suitable for FracAdapt backend). Resolution: 768×768 px from 256×256 input.

**With Test-Time Augmentation (recommended):**

```bash
python inference_v2.py \
  --checkpoint outputs_v2/checkpoints/best_model.pth \
  --input sample_data/dem_30m_sample.tif \
  --output sample_data/dem_10m_sr.asc \
  --tta \
  --device cpu
```

**GeoTIFF output instead of AAIGrid:**

```bash
python inference_v2.py \
  --checkpoint outputs_v2/checkpoints/best_model.pth \
  --input sample_data/dem_30m_sample.tif \
  --output sample_data/dem_10m_sr.tif \
  --format geotiff \
  --device cpu
```

---

## 5. Verify

- **AAIGrid (`.asc`):** Open in QGIS/ArcGIS or any tool that reads Arc ASCII Grid.
- **GeoTIFF:** Open in any raster viewer. Confirm resolution is ~3× finer (30m → 10m).
- Check elevation range and hillshade for plausible terrain.

---

## Summary

| Step | Command |
|------|--------|
| Install | `conda create -n dem-sr-v2 python=3.10` → install PyTorch + deps |
| Checkpoint | Download from Releases or use your own → `outputs_v2/checkpoints/best_model.pth` |
| Sample DEM | `python scripts/create_sample_dem.py` |
| Inference | `python inference_v2.py --checkpoint outputs_v2/checkpoints/best_model.pth --input sample_data/dem_30m_sample.tif --output sample_data/dem_10m_sr.asc --tta` |
