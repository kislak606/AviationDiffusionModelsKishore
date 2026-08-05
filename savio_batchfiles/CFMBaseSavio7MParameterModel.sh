#!/bin/bash
# ---------------------------------------------------------------------------
# SLURM job script: CFM (flow matching) training run, ADS-B trajectory
# ---------------------------------------------------------------------------
#SBATCH --job-name=cfm_trajectory_full
#SBATCH --account=ac_mixedav
#SBATCH --partition=savio3_gpu
#SBATCH --qos=gtx2080_gpu3_normal
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=20:00:00
#SBATCH --output=logs/cfm_%j.out
#SBATCH --error=logs/cfm_%j.err

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
mkdir -p logs

module load anaconda3
source activate adsb

python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

# ---------------------------------------------------------------------------
# Training run
# ---------------------------------------------------------------------------
NC_PATH=/global/scratch/users/kishore26/adsb-diffusion/data/trajectories_adsblol_seq86_stage2.nc
OUTPUT_DIR=/global/scratch/users/kishore26/adsb-diffusion/checkpoints_cfm

mkdir -p $OUTPUT_DIR

cd /global/scratch/users/kishore26/adsb-diffusion/AviationDiffusionModelsKishore/training

python trainCFMBaseSAVIO.py \
    --nc_path $NC_PATH \
    --output_dir $OUTPUT_DIR \
    --epochs 100 \
    --batch_size 64

echo "Job finished at $(date)"