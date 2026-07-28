#!/bin/bash
# ---------------------------------------------------------------------------
# SLURM job script: full DDIM/DiT training run, ADS-B trajectory prediction
# ---------------------------------------------------------------------------
#SBATCH --job-name=ddim_trajectory_full
#SBATCH --account=ac_mixedav
#SBATCH --partition=savio4_gpu
#SBATCH --qos=a5k_gpu4_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --output=logs/ddim_%j.out
#SBATCH --error=logs/ddim_%j.err
 
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
OUTPUT_DIR=/global/scratch/users/kishore26/adsb-diffusion/checkpoints
 
mkdir -p $OUTPUT_DIR
 
cd /global/scratch/users/kishore26/adsb-diffusion/AviationDiffusionModelsKishore/training
 
python trainDDIMBaseSAVIO.py \
    --nc_path $NC_PATH \
    --output_dir $OUTPUT_DIR \
    --epochs 100 \
    --batch_size 64
 
echo "Job finished at $(date)"