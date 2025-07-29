# DDIM Inversion
python baseline_ddim_inv.py \
    --model_path THUDM/CogVideoX-5B \
    --prompt_path /work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/horse_f12/vlm_agent.json \
    --latent_path /work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/horse_f12/inversion.pt \
    --num_inference_steps 250 \
    --output_path /work/hdd/benk/cl121/CogVideo/inference/outputs/baseline_ddim_inv/horse_f12_randn