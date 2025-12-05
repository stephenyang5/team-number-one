#!/bin/bash
#SBATCH --job-name=gtvelo
#SBATCH --partition=gpu          # change to your GPU partition
#SBATCH --gres=gpu:1             # 1 GPU
#SBATCH --cpus-per-task=8        # adjust as needed
#SBATCH --mem=64G                # system RAM
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# If your cluster has GPU types with more memory, constrain them:
# SBATCH --constraint=a100        # or h100, v100-32g, etc.
# Or if your site supports it:
# SBATCH --gres=gpu:a100:1        # request a specific GPU model

module load python
source ~/team-number-one/bin/activate  # activate your Python env

# Set PyTorch CUDA memory allocation to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Ensure log dir exists
mkdir -p logs

# Run with 2+ heads; adjust hidden_channels if needed
python src/train.py \
    --graph_path data/blood/blood_graph_velocity.pt \
    --hidden_channels 256 \
    --heads 2 \
    --num_layers 2 \
    --dropout 0.1 \
    --lr 0.001 \
    --epochs 10 \
    --use_edge_attr true \
    --use_timepoint false \
    --output_dir ./checkpoints_lightweight

