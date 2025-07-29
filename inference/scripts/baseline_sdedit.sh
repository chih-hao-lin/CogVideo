# DDIM Inversion
python baseline_sdedit.py \
    --model_path THUDM/CogVideoX-5B \
    --prompt_path /work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/horse_f12/vlm_agent.json \
    --video_path /work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results_f12/horse/horse.mp4 \
    --num_inference_steps 250 \
    --t_0 0.75 \
    --output_path /work/hdd/benk/cl121/CogVideo/inference/outputs/baseline_sdedit/horse_f12_0.75