import argparse
import math
import os
import random
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union, cast
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from diffusers.models.attention_processor import Attention, CogVideoXAttnProcessor2_0
from diffusers.models.autoencoders import AutoencoderKLCogVideoX
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.transformers.cogvideox_transformer_3d import (
    CogVideoXBlock,
    CogVideoXTransformer3DModel,
)
from diffusers.pipelines.cogvideo.pipeline_cogvideox import CogVideoXPipeline, retrieve_timesteps
from diffusers.schedulers import CogVideoXDDIMScheduler, DDIMInverseScheduler
from diffusers.utils import export_to_video

# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error.
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort: skip
import json
from ddim_inversion import (
    get_video_frames,
    encode_video_frames,
    export_latents_to_video,
    sample,
)
from dynvfx_utils import (
    DynVFXAttnProcessor,
    OverrideAttnProcessors,
    get_latents_mask,
    get_tokens_mask,
    apply_anchor_extended_attention_dropout,
    sample_anchor_extended_attention,
)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path of the pretrained model"
    )
    parser.add_argument(
        "--prompt_path", type=str, required=True, help="Path of the prompt"
    )
    parser.add_argument(
        "--latent_path", type=str, required=True, help="Path of the latent"
    )
    parser.add_argument(
        "--video_path", type=str, required=True, help="Path of the input video"
    )
    parser.add_argument(
        "--mask_orig_path", type=str, required=True, help="Path of the O_orig mask"
    )
    parser.add_argument(
        "--video_edit_path", type=str, default=None, help="Path of the edited video (previous output)"
    )
    parser.add_argument(
        "--mask_edit_path", type=str, default=None, help="Path of the O_edit mask"
    )
    parser.add_argument(
        "--output_path", type=str, default="output", help="Path of the output videos"
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=6.0, help="Classifier-free guidance scale"
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=250, help="Number of inference steps"
    )
    parser.add_argument(
        "--skip_frames_start", type=int, default=0, help="Number of skipped frames from the start"
    )
    parser.add_argument(
        "--skip_frames_end", type=int, default=0, help="Number of skipped frames from the end"
    )
    parser.add_argument(
        "--frame_sample_step", type=int, default=None, help="Temporal stride of the sampled frames"
    )
    parser.add_argument(
        "--max_num_frames", type=int, default=81, help="Max number of sampled frames"
    )
    parser.add_argument(
        "--width", type=int, default=720, help="Resized width of the video frames"
    )
    parser.add_argument(
        "--height", type=int, default=480, help="Resized height of the video frames"
    )
    parser.add_argument(
        "--fps", type=int, default=8, help="Frame rate of the output videos"
    )
    parser.add_argument(
        "--dtype", type=str, default="bf16", choices=["bf16", "fp16"], help="Dtype of the model"
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for the random number generator")
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device for inference"
    )
    parser.add_argument(
        "--t_0", type=float, default=0.8, help="Noise level in (0, 1)"
    )
    parser.add_argument(
        "--aea_dropout_fg", type=float, default=1.0, help="Dropout of the Anchor Extended Attention foreground"
    )
    parser.add_argument(
        "--aea_dropout_bg", type=float, default=0.0, help="Dropout of the Anchor Extended Attention background"
    )
    args = parser.parse_args()
    return args

@torch.no_grad()
def main():
    args = get_args()
    set_seed(args.seed)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = torch.device(args.device)
    dir_output = os.path.dirname(args.output_path)
    os.makedirs(dir_output, exist_ok=True)

    # Load prompt
    with open(args.prompt_path, "r") as f:
        prompt_dict = json.load(f)
    prompt = prompt_dict["composited_scene_caption"]
    prompt_save_path = os.path.join(dir_output, "prompt.txt")
    with open(prompt_save_path, "w") as f:
        f.write(prompt)

    # Load pipeline
    pipeline = CogVideoXPipeline.from_pretrained(
        args.model_path, torch_dtype=dtype
    ).to(device=device)

    # Load video frames
    video_frames = get_video_frames(
        video_path=args.video_path,
        width=args.width,
        height=args.height,
        skip_frames_start=args.skip_frames_start,
        skip_frames_end=args.skip_frames_end,
        max_num_frames=args.max_num_frames,
        frame_sample_step=args.frame_sample_step,
    ).to(device=device)
    video_latents = encode_video_frames(vae=pipeline.vae, video_frames=video_frames)

    # Load mask frames
    mask_orig_frames = get_video_frames(
        video_path=args.mask_orig_path,
        width=args.width,
        height=args.height,
        skip_frames_start=args.skip_frames_start,
        skip_frames_end=args.skip_frames_end,
        max_num_frames=args.max_num_frames,
        frame_sample_step=args.frame_sample_step,
    ).to(device=device)

    # Estimate and apply residual x_res
    if args.video_edit_path != None and args.mask_edit_path != None:
        print("Applying residual x_res")
        video_edit_frames = get_video_frames(
            video_path=args.video_edit_path,
            width=args.width,
            height=args.height,
            skip_frames_start=args.skip_frames_start,
            skip_frames_end=args.skip_frames_end,
            max_num_frames=args.max_num_frames,
            frame_sample_step=args.frame_sample_step,
        ).to(device=device)
        video_edit_latents = encode_video_frames(vae=pipeline.vae, video_frames=video_edit_frames)

        mask_edit_frames = get_video_frames(
            video_path=args.mask_edit_path,
            width=args.width,
            height=args.height,
            skip_frames_start=args.skip_frames_start,
            skip_frames_end=args.skip_frames_end,
            max_num_frames=args.max_num_frames,
            frame_sample_step=args.frame_sample_step,
        ).to(device=device)
        latent_mask_edit = get_latents_mask(
            mask_frames=mask_edit_frames,
            pipeline=pipeline,
        ).unsqueeze(2).to(video_latents.dtype)
        
        video_latents = video_latents + latent_mask_edit * (video_edit_latents - video_latents)

    # Get timesteps and alphas
    scheduler = pipeline.scheduler    
    timesteps, num_inference_steps = retrieve_timesteps(scheduler, args.num_inference_steps, device)
    alphas = pipeline.scheduler.alphas_cumprod[timesteps.cpu()]

    # Get alphas at t_0
    last_n_steps = int(args.t_0 * num_inference_steps)
    alpha_t_0 = alphas[-last_n_steps].to(device)

    # Add noise
    noise = torch.randn_like(video_latents)
    noisy_video_latents = video_latents * alpha_t_0.sqrt() + noise * (1 - alpha_t_0).sqrt()

    inverse_latents = torch.load(args.latent_path).to(device)

    # Denoise
    with OverrideAttnProcessors(pipeline.transformer):
        denoised_latents = sample_anchor_extended_attention(
            pipeline=pipeline,
            latents=noisy_video_latents,
            mask_frames=mask_orig_frames,
            scheduler=pipeline.scheduler,
            last_n_steps=last_n_steps,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=torch.Generator(device=device).manual_seed(args.seed),
            reference_latents=reversed(inverse_latents),
            aea_dropout_fg=args.aea_dropout_fg,
            aea_dropout_bg=args.aea_dropout_bg,
        )

    # Save denoised video
    export_latents_to_video(pipeline, denoised_latents, args.output_path, fps=args.fps)

if __name__ == "__main__":
    main()