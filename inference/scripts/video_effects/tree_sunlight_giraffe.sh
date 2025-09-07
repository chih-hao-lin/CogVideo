#!/bin/sh
#
#SBATCH --job-name=tree_sunlight_giraffe
#SBATCH --output=/work/hdd/benk/cl121/CogVideo/inference/outputs/logs/%j_tree_sunlight_giraffe.out
#SBATCH --error=/work/hdd/benk/cl121/CogVideo/inference/outputs/logs/%j_tree_sunlight_giraffe.err
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

# Set the directory paths
MODEL_PATH=THUDM/CogVideoX-5B
DIR_SAM=/work/hdd/benk/cl121/EVF-SAM
DIR_DYNVFX=/work/hdd/benk/cl121/CogVideo/inference
DIR_OUTPUT=$DIR_DYNVFX/outputs/dynvfx_sem/tree_sunlight_giraffe
PATH_PROMPT=$DIR_OUTPUT/vlm_agent.json
PATH_LATENT=$DIR_OUTPUT/inversion.pt
PATH_VIDEO=/work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results_f9/tree_sunlight/tree_sunlight.mp4
PROMPT="Add a giraffe walking"
PROMPT_ORIG="[semantic] trees"
PROMPT_EDIT="[semantic] giraffe"
PATH_MASK_ORIG=$DIR_OUTPUT/mask_orig.mp4
SEED=0
DIR_SAMPLE=$DIR_OUTPUT/sampling_0_seed_$SEED
mkdir -p $DIR_SAMPLE
cp $DIR_DYNVFX/scripts/video_effects/tree_sunlight_giraffe.sh $DIR_SAMPLE
NUM_SAMPLES=250

# id, t_0, aea_dropout_fg, aea_dropout_bg
params=(
    "0 0.9 0.3 0.2"
    "1 0.8 0.3 0.2"
    "2 0.7 0.3 0.2"
    "3 0.5 0.3 0.2"
    "4 0.3 0.3 0.2"
    "5 0.1 0.3 0.2"
)

# VLM Agent
python vlm_agent.py \
    --vfx_prompt "$PROMPT" \
    --num_key_frames 5 \
    --video_path $PATH_VIDEO \
    --output_path $PATH_PROMPT

# Mask estimation for PROMPT_ORIG
cd $DIR_SAM
python inference_video.py  \
    --version YxZhang/evf-sam2 \
    --precision='fp16' \
    --model_type sam2   \
    --video_path $PATH_VIDEO \
    --vis_save_path $PATH_MASK_ORIG \
    --prompt "$PROMPT_ORIG"
cd $DIR_DYNVFX

# DDIM Inversion
python ddim_inversion.py \
    --model_path $MODEL_PATH \
    --prompt "" \
    --save_latents --num_inference_steps $NUM_SAMPLES \
    --video_path  $PATH_VIDEO \
    --output_path $DIR_OUTPUT

# Iterative sampling
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
            --model_path          $MODEL_PATH \
            --t_0                 $t_0 \
            --aea_dropout_fg      $aea_dropout_fg \
            --aea_dropout_bg      $aea_dropout_bg \
            --prompt_path         $PATH_PROMPT \
            --latent_path         $PATH_LATENT \
            --video_path          $PATH_VIDEO \
            --mask_orig_path      $PATH_MASK_ORIG \
            --num_inference_steps $NUM_SAMPLES \
            --seed                $SEED \
            --output_path         $DIR_SAMPLE/output_$id.mp4
    else
        PATH_VIDEO_EDIT=$DIR_SAMPLE/output_$(($id-1)).mp4
        PATH_MASK_EDIT=$DIR_SAMPLE/mask_edit_$(($id-1)).mp4
        python dynvfx.py \
            --model_path          $MODEL_PATH \
            --t_0                 $t_0 \
            --aea_dropout_fg      $aea_dropout_fg \
            --aea_dropout_bg      $aea_dropout_bg \
            --prompt_path         $PATH_PROMPT \
            --latent_path         $PATH_LATENT \
            --video_path          $PATH_VIDEO \
            --mask_orig_path      $PATH_MASK_ORIG \
            --video_edit_path     $PATH_VIDEO_EDIT \
            --mask_edit_path      $PATH_MASK_EDIT \
            --num_inference_steps $NUM_SAMPLES \
            --seed                $SEED \
            --output_path         $DIR_SAMPLE/output_$id.mp4
    fi

    # Run segmentation
    cd $DIR_SAM
    python inference_video.py \
        --version YxZhang/evf-sam2 \
        --precision='fp16' \
        --model_type sam2 \
        --video_path $DIR_SAMPLE/output_$id.mp4 \
        --vis_save_path $DIR_SAMPLE/mask_edit_$id.mp4 \
        --prompt "$PROMPT_EDIT"
    cd $DIR_DYNVFX
done
