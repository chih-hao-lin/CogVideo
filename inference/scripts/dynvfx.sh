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
t_0=1.0
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