#!/bin/bash
# From repo root: bash testing/run_example.sh
set -e
cd "$(dirname "$0")/.."
CHECKPOINT="outputs_v2/checkpoints/best_model.pth"
INPUT="testing/sample_data/dem_30m_sample.tif"
OUTPUT="testing/sample_data/dem_10m_sr.asc"
[[ -f "$CHECKPOINT" ]] || { echo "Missing $CHECKPOINT - see testing/README.md"; exit 1; }
[[ -f "$INPUT" ]] || { echo "Missing $INPUT - run: python testing/create_sample_dem.py"; exit 1; }
python inference_v2.py --checkpoint "$CHECKPOINT" --input "$INPUT" --output "$OUTPUT" --format aaigrid --tta --device cpu
echo "Done: $OUTPUT"
