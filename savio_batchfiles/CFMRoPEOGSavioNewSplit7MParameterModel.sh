#!/bin/bash
#SBATCH --job-name=cfm_rope_og_newsplit_s0
#SBATCH --account=fc_mixedav
#SBATCH --partition=savio3_gpu
#SBATCH --qos=gtx2080_gpu3_normal
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=20:00:00
#SBATCH --output=logs/cfm_rope_og_%j.out
#SBATCH --error=logs/cfm_rope_og_%j.err

# run `mkdir -p logs` BEFORE sbatch; Slurm needs it to exist at submit time

module load anaconda3
source activate adsb

python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('Device:', torch.cuda.get_device_name(0))" || { echo "CUDA check failed, aborting"; exit 1; }

NC_PATH=/global/scratch/users/kishore26/adsb-diffusion/data/trajectories_adsblol_seq86_stage2.nc
OUTPUT_DIR=$HOME/checkpoints_cfm_rope_og_newSplit_s0
mkdir -p $OUTPUT_DIR

cd $HOME/AviationDiffusionModelsKishore/training

python -u trainCFMRoPEOGNewSplitSavio.py \
    --nc_path $NC_PATH \
    --output_dir $OUTPUT_DIR \
    --epochs 100 \
    --batch_size 64 \
    --seed 0

echo "Job finished at $(date)"