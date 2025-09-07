# python cli_demo.py \
#     --model_path THUDM/CogVideoX-5B --generate_type "t2v" \
#     --output_path ./outputs/test/t2v_girl.mp4 \
#     --prompt "A young girl with flowing hair rides a vintage bicycle along a sun-dappled path, surrounded by vibrant wildflowers swaying gently in the breeze. She wears a light summer dress, its fabric fluttering as she pedals with joyful abandon. Her wicker basket, attached to the handlebars, overflows with freshly picked daisies, adding a touch of whimsy to her journey. The golden sunlight casts playful shadows on the path, highlighting her carefree spirit. As she rides, the distant sound of birds chirping and the rustle of leaves create a serene soundtrack, capturing the essence of a perfect summer day." 

DIR_SAM=/work/hdd/benk/cl121/EVF-SAM
DIR_DYNVFX=/work/hdd/benk/cl121/CogVideo/inference

cd $DIR_SAM
python inference_video.py \
    --version YxZhang/evf-sam2 \
    --precision='fp16' \
    --model_type sam2 \
    --video_path    /work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/creek_deer/sampling_1_seed_0/output_3.mp4 \
    --vis_save_path /work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/creek_deer/sampling_1_seed_0/mask_edit_3_test_semantic.mp4 \
    --prompt "[semantic] deers"
cd $DIR_DYNVFX