#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_minimal_stego_experiment.sh \
    GENOME_MANIFEST IMAGE_MANIFEST OUT_ROOT KEY_FILE

Optional environment variables:
  N_TRAIN=400 N_VAL=100 N_TEST=200 EPOCHS=3 SEED=2026
  PHYSICAL_GPU=3 WORKERS=4 RUN_CNN=1 RUN_HYENA=0
  HYENA_MODEL=/absolute/path/to/hyenadna-medium-160k-hf
  REUSE_DATASET=0
EOF
}

if [[ "$#" -ne 4 ]]; then
  usage >&2
  exit 2
fi

GENOME_MANIFEST=$1
IMAGE_MANIFEST=$2
OUT_ROOT=$3
KEY_FILE=$4
N_TRAIN=${N_TRAIN:-400}
N_VAL=${N_VAL:-100}
N_TEST=${N_TEST:-200}
EPOCHS=${EPOCHS:-3}
SEED=${SEED:-2026}
PHYSICAL_GPU=${PHYSICAL_GPU:-3}
WORKERS=${WORKERS:-4}
RUN_CNN=${RUN_CNN:-1}
RUN_HYENA=${RUN_HYENA:-0}
REUSE_DATASET=${REUSE_DATASET:-0}
DATASET="$OUT_ROOT/dataset_l0_l3"

for path in "$GENOME_MANIFEST" "$IMAGE_MANIFEST" "$KEY_FILE"; do
  if [[ ! -s "$path" ]]; then
    echo "Required input is missing or empty: $path" >&2
    exit 3
  fi
done

mkdir -p "$OUT_ROOT/logs"
cat > "$OUT_ROOT/run_config.txt" <<EOF
genome_manifest=$GENOME_MANIFEST
image_manifest=$IMAGE_MANIFEST
key_file=$KEY_FILE
n_train=$N_TRAIN
n_val=$N_VAL
n_test=$N_TEST
epochs=$EPOCHS
seed=$SEED
physical_gpu=$PHYSICAL_GPU
workers=$WORKERS
run_cnn=$RUN_CNN
run_hyena=$RUN_HYENA
hyena_model=${HYENA_MODEL:-}
EOF

if [[ -e "$DATASET/dataset_manifest.json" && "$REUSE_DATASET" != "1" ]]; then
  echo "Dataset already exists: $DATASET" >&2
  echo "Set REUSE_DATASET=1 to audit and train from it, or choose a new OUT_ROOT." >&2
  exit 4
fi

if [[ ! -e "$DATASET/dataset_manifest.json" ]]; then
  python -u -m genome_mining build-image-dataset \
    --genome-manifest "$GENOME_MANIFEST" \
    --image-manifest "$IMAGE_MANIFEST" \
    --out "$DATASET" \
    --n-train "$N_TRAIN" --n-val "$N_VAL" --n-test "$N_TEST" \
    --positive-fraction 0.5 \
    --image-size 128 --resize-short-side 256 \
    --window-length 131072 \
    --codecs direct encrypted constrained kmer \
    --codec-key-file "$KEY_FILE" \
    --kmer-order 3 \
    --seed "$SEED" \
    2>&1 | tee "$OUT_ROOT/logs/build_dataset.log"
fi

python -u -m genome_mining audit-image-dataset \
  --dataset "$DATASET" \
  --out "$OUT_ROOT/dataset_audit.json" \
  2>&1 | tee "$OUT_ROOT/logs/audit.log"

python -u -m genome_mining verify-stego-roundtrip \
  --dataset "$DATASET" \
  --codec-key-file "$KEY_FILE" \
  --out "$OUT_ROOT/oracle_roundtrip" \
  2>&1 | tee "$OUT_ROOT/logs/oracle_roundtrip.log"

python -u -m genome_mining summarize-stego-dataset \
  --dataset "$DATASET" \
  --out "$OUT_ROOT/stego_summary" \
  2>&1 | tee "$OUT_ROOT/logs/stego_summary.log"

export CUDA_VISIBLE_DEVICES=$PHYSICAL_GPU

if [[ "$RUN_CNN" == "1" ]]; then
  python -u -m genome_mining train-image-locator \
    --dataset "$DATASET" \
    --out "$OUT_ROOT/cnn" \
    --backbone cnn \
    --epochs "$EPOCHS" --batch-size 4 \
    --lr 2e-4 --d-model 128 --downsample-stride 256 \
    --context dilated_cnn --context-layers 6 \
    --workers "$WORKERS" --device cuda --amp \
    --codec-key-file "$KEY_FILE" \
    --seed "$SEED" \
    2>&1 | tee "$OUT_ROOT/logs/train_cnn.log"
fi

if [[ "$RUN_HYENA" == "1" ]]; then
  if [[ -z "${HYENA_MODEL:-}" || ! -d "$HYENA_MODEL" ]]; then
    echo "RUN_HYENA=1 requires HYENA_MODEL to be an existing local model directory." >&2
    exit 5
  fi
  python -u -m genome_mining train-image-locator \
    --dataset "$DATASET" \
    --out "$OUT_ROOT/hyenadna_rc" \
    --backbone hyenadna \
    --hyena-model-name "$HYENA_MODEL" --hyena-local-files-only \
    --hyena-fine-tune frozen \
    --epochs "$EPOCHS" --batch-size 1 --gradient-accumulation 8 \
    --lr 2e-4 --d-model 128 --downsample-stride 256 \
    --workers "$WORKERS" --device cuda --amp \
    --codec-key-file "$KEY_FILE" \
    --seed "$SEED" \
    2>&1 | tee "$OUT_ROOT/logs/train_hyenadna.log"
fi

echo "Completed minimal stego experiment."
echo "Dataset audit: $OUT_ROOT/dataset_audit.json"
echo "Oracle recovery: $OUT_ROOT/oracle_roundtrip/roundtrip_report.json"
echo "Stego summary: $OUT_ROOT/stego_summary/stego_report.csv"
if [[ "$RUN_CNN" == "1" ]]; then
  echo "CNN metrics: $OUT_ROOT/cnn/test_evaluation/metrics.json"
fi
if [[ "$RUN_HYENA" == "1" ]]; then
  echo "HyenaDNA metrics: $OUT_ROOT/hyenadna_rc/test_evaluation/metrics.json"
fi
