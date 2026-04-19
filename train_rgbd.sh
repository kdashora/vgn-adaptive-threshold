#!/bin/bash
#SBATCH --job-name=vgn_rgbd_train
#SBATCH --output=logs/vgn_rgbd_%j.out
#SBATCH --error=logs/vgn_rgbd_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=htc

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "Start:  $(date)"
echo "=========================================="

# ── Environment ───────────────────────────────
source ~/miniconda3/bin/activate
conda activate vgn
cd ~/vgn

# ── Generate full RGB-D dataset on SOL ────────
# Step 1: Generate raw grasp data with RGB images
echo "=== Step 1: Generating raw RGB-D grasp data ==="
mkdir -p data/raw/rgbd_full
mkdir -p logs

python scripts/generate_data.py data/raw/rgbd_full \
    --scene pile \
    --object-set blocks \
    --num-grasps 50000

python scripts/generate_data.py data/raw/rgbd_full \
    --scene packed \
    --object-set blocks \
    --num-grasps 50000

echo "=== Step 1 complete: $(date) ==="

# ── Step 2: Construct 4-channel voxel dataset ─
echo "=== Step 2: Constructing RGB-D voxel grids ==="
mkdir -p data/datasets/rgbd_full

python scripts/construct_dataset.py \
    data/raw/rgbd_full \
    data/datasets/rgbd_full

echo "=== Step 2 complete: $(date) ==="

# ── Step 3: Train the RGB-D network ───────────
echo "=== Step 3: Training RGB-D VGN ==="
mkdir -p data/runs

python scripts/train_vgn.py \
    --dataset data/datasets/rgbd_full \
    --net conv \
    --epochs 30 \
    --batch-size 32 \
    --lr 3e-4 \
    --val-split 0.1 \
    --augment \
    --description rgbd_fusion

echo "=== Training complete: $(date) ==="
echo "=========================================="
