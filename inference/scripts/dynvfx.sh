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
cd /work/hdd/benk/cl121/EVF-SAM
python inference_video.py  \
  --version YxZhang/evf-sam2 \
  --precision='fp16' \
  --vis_save_path "/work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/horse_f12/mask" \
  --model_type sam2   \
  --video_path "/work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results_f12/horse/horse.mp4" \
  --prompt "horse"
cd /work/hdd/benk/cl121/CogVideo/inference