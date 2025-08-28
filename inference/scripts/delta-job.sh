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

DIR_SAM=/work/hdd/benk/cl121/EVF-SAM
DIR_DYNVFX=/work/hdd/benk/cl121/CogVideo/inference
DIR_OUTPUT=$DIR_DYNVFX/outputs/dynvfx/horse_f12
PATH_PROMPT=$DIR_OUTPUT/vlm_agent.json
PATH_LATENT=$DIR_OUTPUT/inversion.pt
PATH_VIDEO=/work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results_f12/horse/horse.mp4
PROMPT_ORIG="horse"
PROMPT_EDIT="knight"
PATH_MASK_ORIG=$DIR_OUTPUT/mask_orig.mp4
SEED=0

# id, t_0, aea_dropout_fg, aea_dropout_bg
params=(
    "0 1.0 0.3 0.2"
    "1 0.9 0.3 0.2"
    "2 0.7 0.3 0.2"
    "3 0.5 0.3 0.2"
    "4 0.3 0.0 0.0"
)

DIR_SAVE=$DIR_OUTPUT/test_iterative_4_seed_$SEED
mkdir -p $DIR_SAVE
cp $DIR_DYNVFX/scripts/delta-job.sh $DIR_SAVE

for param in "${params[@]}"; do
    set -- $param
    id=$1
    t_0=$2
    aea_dropout_fg=$3
    aea_dropout_bg=$4
    echo "Running $id: t_0=$t_0, aea_dropout_fg=$aea_dropout_fg, aea_dropout_bg=$aea_dropout_bg"

    # if id ==0, set $PATH_MASK_EDIT to none, else set to $PATH_MASK_EDIT
    if [ $id -eq 0 ]; then
        python dynvfx.py \
            --model_path THUDM/CogVideoX-5B \
            --t_0               $t_0 \
            --aea_dropout_fg    $aea_dropout_fg \
            --aea_dropout_bg    $aea_dropout_bg \
            --prompt_path       $PATH_PROMPT \
            --latent_path       $PATH_LATENT \
            --video_path        $PATH_VIDEO \
            --mask_orig_path    $PATH_MASK_ORIG \
            --seed              $SEED \
            --output_path       $DIR_SAVE/output_$id.mp4
    else
        PATH_VIDEO_EDIT=$DIR_SAVE/output_$(($id-1)).mp4
        PATH_MASK_EDIT=$DIR_SAVE/mask_edit_$(($id-1)).mp4
        python dynvfx.py \
            --model_path THUDM/CogVideoX-5B \
            --t_0               $t_0 \
            --aea_dropout_fg    $aea_dropout_fg \
            --aea_dropout_bg    $aea_dropout_bg \
            --prompt_path       $PATH_PROMPT \
            --latent_path       $PATH_LATENT \
            --video_path        $PATH_VIDEO \
            --mask_orig_path    $PATH_MASK_ORIG \
            --video_edit_path   $PATH_VIDEO_EDIT \
            --mask_edit_path    $PATH_MASK_EDIT \
            --seed              $SEED \
            --output_path       $DIR_SAVE/output_$id.mp4
    fi

    # Run segmentation
    cd $DIR_SAM
    python inference_video.py \
        --version YxZhang/evf-sam2 \
        --precision='fp16' \
        --model_type sam2 \
        --video_path $DIR_SAVE/output_$id.mp4 \
        --vis_save_path $DIR_SAVE/mask_edit_$id.mp4 \
        --prompt $PROMPT_EDIT
    cd $DIR_DYNVFX
done