import argparse
import math
import os
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union, cast

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
    CogVideoXAttnProcessor2_0ForDDIMInversion,
    OverrideAttnProcessors,
    get_video_frames,
    encode_video_frames,
    export_latents_to_video,
    sample,
)

@torch.no_grad()
def main():
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
        "--output_path", type=str, default="output", help="Path of the output videos"
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=6.0, help="Classifier-free guidance scale"
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="Number of inference steps"
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
    parser.add_argument("--width", type=int, default=720, help="Resized width of the video frames")
    parser.add_argument(
        "--height", type=int, default=480, help="Resized height of the video frames"
    )
    parser.add_argument("--fps", type=int, default=8, help="Frame rate of the output videos")
    parser.add_argument(
        "--dtype", type=str, default="bf16", choices=["bf16", "fp16"], help="Dtype of the model"
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for the random number generator")
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device for inference"
    )
    parser.add_argument("--save_latents", action="store_true", help="Save the latents")
    args = parser.parse_args()
    args.dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    args.device = torch.device(args.device)
    os.makedirs(args.output_path, exist_ok=True)

    with open(args.prompt_path, "r") as f:
        prompt_dict = json.load(f)
    prompt = prompt_dict["composited_scene_caption"]
    prompt_save_path = os.path.join(args.output_path, "prompt.txt")
    with open(prompt_save_path, "w") as f:
        f.write(prompt)

    pipeline = CogVideoXPipeline.from_pretrained(
        args.model_path, torch_dtype=args.dtype
    ).to(device=args.device)

    inverse_latents = torch.load(args.latent_path).to(args.device)
    with OverrideAttnProcessors(transformer=pipeline.transformer):
        recon_latents = sample(
            pipeline=pipeline,
            latents=torch.randn_like(inverse_latents[-1]),
            # latents=inverse_latents[-1], # use the last latents as the initial latents
            scheduler=pipeline.scheduler,
            prompt=prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=torch.Generator(device=args.device).manual_seed(args.seed),
            reference_latents=reversed(inverse_latents),
        )

    # recon_latents = sample(
    #     pipeline=pipeline,
    #     latents=torch.randn_like(inverse_latents[-1]),
    #     # latents=inverse_latents[-1], # use the last latents as the initial latents
    #     scheduler=pipeline.scheduler,
    #     prompt=prompt,
    #     num_inference_steps=args.num_inference_steps,
    #     guidance_scale=args.guidance_scale,
    #     generator=torch.Generator(device=args.device).manual_seed(args.seed),
    #     reference_latents=None, #reversed(inverse_latents),
    # )
    
    
    video_path = os.path.join(args.output_path, "video.mp4")
    export_latents_to_video(pipeline, recon_latents[-1], video_path, args.fps)

if __name__ == "__main__":
    main()