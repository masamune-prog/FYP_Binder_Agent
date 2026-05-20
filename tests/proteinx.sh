#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "========================================"
echo "🧪 Starting Protenix Integration Test"
echo "========================================"

# 1. Define paths and variables
TEST_DIR="./protenix_test_run"
INPUT_JSON="$TEST_DIR/test_input.json"
OUT_DIR="$TEST_DIR/output"
MODEL_NAME="protenix_base_default_v0.5.0"

# 2. Clean up previous test runs and recreate directories
echo "[1/3] Setting up test directories..."
rm -rf "$TEST_DIR"
mkdir -p "$OUT_DIR"

# 3. Create a minimal dummy JSON payload
# Using a very short sequence (MAAGVK) so the inference completes quickly
echo "[2/3] Generating minimal JSON input at $INPUT_JSON..."
cat <<EOF > "$INPUT_JSON"
[
  {
    "name": "protenix_dummy_test",
    "modelSeeds": [1],
    "sequences": [
      {
        "proteinChain": {
          "sequence": "MAAGVK",
          "count": 1
        }
      }
    ]
  }
]
EOF

# 4. Run the Protenix command
echo "[3/3] Executing Protenix prediction..."
echo "Command: protenix predict --input $INPUT_JSON --out_dir $OUT_DIR --model_name $MODEL_NAME"
echo "----------------------------------------"

protenix predict \
    --input "$INPUT_JSON" \
    --out_dir "$OUT_DIR" \
    --model_name "$MODEL_NAME"

echo "----------------------------------------"
echo "✅ Test completed successfully!"
echo "Check the output directory for results: $OUT_DIR"