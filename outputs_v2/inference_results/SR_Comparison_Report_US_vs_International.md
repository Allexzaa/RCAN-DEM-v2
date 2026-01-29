# DEM-SR v2: Comparison Report — US Ground-Truth vs International Results

**Model:** DEM-SR v2 (RCAN-DEM v2)  
**Checkpoint:** Best model (e.g. epoch 189, val loss 0.6236)  
**Purpose:** Compare 10m super-resolution results where **real 10m ground truth exists** (US, USGS 3DEP) with **international regions** where only 30m input is available (reconstruction fidelity only).

---

## 1. Executive Summary

- **International test** (Afghanistan, Iraq, Vietnam): No 10m reference DEM exists. Quality is assessed via **reconstruction fidelity** — downsampling the 10m SR back to 30m and comparing to the original 30m input (RMSE, MAE, correlation, mean elevation difference).
- **US ground-truth test** (Kansas, Colorado, Appalachian): Paired 30m (Copernicus GLO-30) and 10m (USGS 3DEP) data allow **direct comparison** of 10m SR output to real 10m reference, plus the same reconstruction-fidelity metrics for consistency with international.

This report presents both sets of results and compares them so that:
1. **Reconstruction fidelity** (downsampled SR vs 30m input) can be compared across all regions (US and international).
2. **US-only:** True 10m accuracy (SR vs real 10m) shows how well the model approximates real high-resolution terrain when reference data exists.

---

## 2. Methodology

### 2.1 International test (no 10m ground truth)

- **Regions:** Afghanistan (Kabul), Iraq (Baghdad), Vietnam (Hanoi) — 0.25° × 0.25° each.
- **Input:** 30m Copernicus GLO-30 GeoTIFF.
- **Output:** 10m SR in AAIGrid format (TTA, sliding window).
- **Metrics:** Elevation statistics; **reconstruction fidelity** (downsampled 10m SR vs 30m input: RMSE, MAE, correlation, mean elevation difference); gradient statistics.

### 2.2 US ground-truth test (with real 10m reference)

- **Regions:** Kansas (flat), Colorado (steep mountains), Appalachian (moderate hills) — 0.25° × 0.25° clips from full regional DEMs.
- **Input:** 30m Copernicus GLO-30 (clipped to 0.25°).
- **Reference:** 10m USGS 3DEP (clipped to same 0.25°).
- **Output:** 10m SR in AAIGrid format (same pipeline as international).
- **Metrics:**
  - **Vs ground truth 10m:** RMSE (m), MAE (m), Pearson correlation (SR vs real 10m).
  - **Reconstruction fidelity:** Same as international (downsampled SR vs 30m input) for direct comparison.
  - **Elevation bias:** Mean(SR) − mean(reference 10m).

**Data requirement:** Paired 30m and 10m US data must be available (e.g. from `DEM-SR-Model/scripts/download_dem_data.py`). Then run `DEM-SR-Model-v2/scripts/run_us_ground_truth_test.py` to clip, run inference, and compute metrics.

---

## 3. International Results (Reconstruction Fidelity Only)

From `international/sr_quality_metrics.json` and the [10m SR Results Report](international/10m_SR_Results_Report.md):

| Region      | Terrain    | RMSE (m) | MAE (m) | Correlation | Mean diff (m) |
|------------|------------|----------|---------|-------------|----------------|
| Afghanistan| Mountainous| 70.52    | 9.80    | 0.915       | −5.78         |
| Iraq       | Flat       | **1.77** | **0.57**| 0.964       | **−0.34**     |
| Vietnam    | Mixed      | 18.57    | 3.56    | **0.993**   | −2.05         |

*Interpretation:* Flat terrain (Iraq) has the best reconstruction agreement; mountainous (Afghanistan) has higher RMSE but still strong correlation; mixed (Vietnam) has the highest correlation. No 10m ground truth is available for these regions.

---

## 4. US Results (With 10m Ground Truth)

US metrics are written to `us_ground_truth/us_ground_truth_metrics.json` after running the US ground-truth test.

### 4.1 How to generate US metrics

1. **Obtain paired US data** (from repository root `DEM-SR-Model`):
   ```bash
   cd DEM-SR-Model
   python scripts/download_dem_data.py --regions kansas colorado appalachian
   ```
   This creates `raw_data/30m/` and `raw_data/10m/` with `{region}_30m.tif` and `{region}_10m.tif`.

2. **Run US ground-truth test** (from `DEM-SR-Model-v2`):
   ```bash
   cd DEM-SR-Model-v2
   conda run -n dem-sr python scripts/run_us_ground_truth_test.py
   ```
   Optional: `--data_dir /path/to/raw_data`, `--checkpoint path/to/best_model.pth`, `--device cuda`.

3. **Recompute metrics only** (if inference outputs already exist):
   ```bash
   conda run -n dem-sr python scripts/compute_us_gt_metrics.py
   ```

### 4.2 US results table (when available)

Once `us_ground_truth/us_ground_truth_metrics.json` exists, fill or update the following from that file.

**Vs ground truth 10m (SR vs USGS 3DEP 10m):**

| Region     | Terrain         | RMSE (m) | MAE (m) | Correlation |
|------------|-----------------|----------|---------|-------------|
| Kansas     | Flat            | —        | —       | —           |
| Colorado   | Steep Mountains | —        | —       | —           |
| Appalachian| Moderate Hills  | —        | —       | —           |

**Reconstruction fidelity (downsampled SR vs 30m input) — US:**

| Region     | RMSE (m) | MAE (m) | Correlation | Mean diff (m) |
|------------|----------|---------|-------------|----------------|
| Kansas     | —        | —       | —           | —              |
| Colorado   | —        | —       | —           | —              |
| Appalachian| —        | —       | —           | —              |

*If the JSON file is missing or regions failed (e.g. missing `raw_data`), run the steps in §4.1. Once `us_ground_truth/us_ground_truth_metrics.json` contains successful runs, copy the values from each region’s `vs_ground_truth_10m` and `reconstruction_fidelity` into the tables above.*

---

## 5. Side-by-Side Comparison

### 5.1 Reconstruction fidelity: International vs US

Comparing **downsampled 10m SR vs 30m input** across all regions shows whether the model behaves consistently:

- **International:** Iraq (flat) best RMSE/MAE; Vietnam highest correlation; Afghanistan (mountainous) higher RMSE, good correlation.
- **US (when run):** Expect flat (Kansas) to show lower RMSE/MAE and high correlation; Colorado (steep) higher RMSE; Appalachian (moderate hills) in between. This would align with the international trend: reconstruction fidelity is terrain-dependent, with flat terrain agreeing best.

### 5.2 What ground truth adds (US only)

- **Vs ground truth 10m:** RMSE/MAE and correlation of SR vs real 10m quantify **true 10m accuracy** (not just consistency with 30m input).
- **Interpretation:** If reconstruction fidelity is strong (high correlation, low mean diff) but vs-GT RMSE is higher, the model is adding 10m-scale detail that is plausible relative to the 30m input but may differ from the specific 10m product (e.g. different acquisition or processing). If vs-GT correlation is high and RMSE is low, the SR is close to the reference 10m DEM.

---

## 6. Conclusions

1. **International:** Assessment is limited to reconstruction fidelity and elevation/gradient statistics; no 10m reference exists. Results are consistent with terrain type (flat best, mountainous higher RMSE, high correlations overall).
2. **US ground-truth test:** Provides direct 10m accuracy (SR vs USGS 3DEP) and the same reconstruction-fidelity metrics for comparison with international.
3. **Comparison:** Placing US reconstruction-fidelity metrics alongside international allows a consistent view of how well the model preserves the 30m signal across terrain types. US vs-GT metrics then show how well that translates to agreement with real 10m data where available.

---

## 7. Files and Reproducibility

| Item | Location |
|------|----------|
| International metrics | `outputs_v2/inference_results/international/sr_quality_metrics.json` |
| International report | `outputs_v2/inference_results/international/10m_SR_Results_Report.md` |
| US ground-truth metrics | `outputs_v2/inference_results/us_ground_truth/us_ground_truth_metrics.json` |
| US test script | `DEM-SR-Model-v2/scripts/run_us_ground_truth_test.py` |
| US metrics-only script | `DEM-SR-Model-v2/scripts/compute_us_gt_metrics.py` |
| US data download | `DEM-SR-Model/scripts/download_dem_data.py` |

To regenerate US metrics after updating data or model:

```bash
cd DEM-SR-Model-v2
conda activate dem-sr
python scripts/run_us_ground_truth_test.py
# Or metrics only:
python scripts/compute_us_gt_metrics.py
```

---

*Report generated for DEM-SR v2 comparison of US ground-truth vs international (reconstruction-only) results.*
