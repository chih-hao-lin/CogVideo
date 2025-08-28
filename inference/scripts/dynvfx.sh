# prompt optimization
# python convert_demo.py --prompt "A man and a woman sit on a sofa, and open a box" --type "i2v" \
#     --image_path /work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results_img/box/box.png

# # DDIM Inversion
# python ddim_inversion.py \
#     --model_path THUDM/CogVideoX-5B \
#     --prompt "" \
#     --save_latents --num_inference_steps 250 \
#     --video_path  /work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results_f12/horse/horse.mp4 \
#     --output_path /work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/horse_f12

# # VLM Agent
# python vlm_agent.py \
#     --vfx_prompt "add a majestic knight riding the horse!" \
#     --num_key_frames 5 \
#     --video_path /work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results_f12/horse/horse.mp4 \
#     --output_path /work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/horse_f12/vlm_agent.json

# Mask estimation
# cd /work/hdd/benk/cl121/EVF-SAM
# python inference_video.py  \
#     --version YxZhang/evf-sam2 \
#     --precision='fp16' \
#     --model_type sam2   \
#     --video_path "/work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results_f12/horse/horse.mp4" \
#     --vis_save_path "/work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/horse_f12/mask_orig.mp4" \
#     --prompt "horse"
# cd /work/hdd/benk/cl121/CogVideo/inference

# Anchor Extended Attention + Content Harmonization
# t_0=1.0
# aea_dropout_fg=0.3
# aea_dropout_bg=0.2
# python dynvfx.py \
#     --model_path THUDM/CogVideoX-5B \
#     --t_0 $t_0 \
#     --aea_dropout_fg $aea_dropout_fg \
#     --aea_dropout_bg $aea_dropout_bg \
#     --prompt_path    /work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/horse_f12/vlm_agent.json \
#     --latent_path    /work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/horse_f12/inversion.pt \
#     --video_path     /work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results_f12/horse/horse.mp4 \
#     --mask_orig_path /work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/horse_f12/mask_orig.mp4 \
#     --output_path    /work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/horse_f12/output/t0-$t_0-aea_fg-$aea_dropout_fg-bg-$aea_dropout_bg.mp4

# cd /work/hdd/benk/cl121/EVF-SAM
# python inference_video.py  \
#     --version YxZhang/evf-sam2 \
#     --precision='fp16' \
#     --model_type sam2   \
#     --video_path    /work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/horse_f12/test_1/output_0.mp4 \
#     --vis_save_path /work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/horse_f12/test_1/mask_edit_0.mp4 \
#     --prompt "knight"
# cd /work/hdd/benk/cl121/CogVideo/inference

# t_0=0.7
# aea_dropout_fg=0.3
# aea_dropout_bg=0.2
# python dynvfx.py \
#     --model_path THUDM/CogVideoX-5B \
#     --t_0 $t_0 \
#     --aea_dropout_fg $aea_dropout_fg \
#     --aea_dropout_bg $aea_dropout_bg \
#     --prompt_path     /work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/horse_f12/vlm_agent.json \
#     --latent_path     /work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/horse_f12/inversion.pt \
#     --video_path      /work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results_f12/horse/horse.mp4 \
#     --mask_orig_path  /work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/horse_f12/mask_orig.mp4 \
#     --video_edit_path /work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/horse_f12/test_1/output_0.mp4 \
#     --mask_edit_path  /work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/horse_f12/test_1/mask_edit_0.mp4 \
#     --output_path     /work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/horse_f12/test_1/output_1.mp4

DIR_SAM=/work/hdd/benk/cl121/EVF-SAM
DIR_DYNVFX=/work/hdd/benk/cl121/CogVideo/inference
DIR_OUTPUT=$DIR_DYNVFX/outputs/dynvfx/horse_f12
PATH_PROMPT=$DIR_OUTPUT/vlm_agent.json
PATH_LATENT=$DIR_OUTPUT/inversion.pt
PATH_VIDEO=/work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results_f12/horse/horse.mp4
PROMPT_ORIG="horse"
PROMPT_EDIT="knight"
PATH_MASK_ORIG=$DIR_OUTPUT/mask_orig.mp4

# id, t_0, aea_dropout_fg, aea_dropout_bg
params=(
    "0 1.0 0.3 0.2"
    "1 0.8 0.3 0.2"
    "2 0.5 0.0 0.0"
)

DIR_SAVE=$DIR_OUTPUT/test_iterative_0
mkdir -p $DIR_SAVE
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