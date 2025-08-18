#!/bin/sh
#
#SBATCH --job-name=dynvfx
#SBATCH --output=/work/hdd/benk/cl121/CogVideo/inference/outputs/logs/%j.out
#SBATCH --error=/work/hdd/benk/cl121/CogVideo/inference/outputs/logs/%j.err
#
#SBATCH --account=bfaf-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --time=2-0:00
#SBATCH --mem=64GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1

cd /work/hdd/benk/cl121/CogVideo/inference
# python dynvfx_utils.py --t_0 0.9 --aea_dropout_fg 0.5 --aea_dropout_bg 0.0

t_0=0.8
aea_dropout_fg=0.3
aea_dropout_bg=0.2
python dynvfx.py \
    --model_path THUDM/CogVideoX-5B \
    --t_0 $t_0 \
    --aea_dropout_fg $aea_dropout_fg \
    --aea_dropout_bg $aea_dropout_bg \
    --prompt_path    /work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/horse_f12/vlm_agent.json \
    --latent_path    /work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/horse_f12/inversion.pt \
    --video_path     /work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results_f12/horse/horse.mp4 \
    --mask_orig_path /work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/horse_f12/mask_orig.mp4 \
    --output_path    /work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/horse_f12/output/t0-$t_0-aea_fg-$aea_dropout_fg-bg-$aea_dropout_bg.mp4