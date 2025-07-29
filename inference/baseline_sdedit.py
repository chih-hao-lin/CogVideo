import argparse
import math
import os
from time import time_ns
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union, cast
import random
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
    CogVideoXAttnProcessor2_0ForDDIMInversion,
    OverrideAttnProcessors,
    get_video_frames,
    encode_video_frames,
    export_latents_to_video,
    sample,
)

# Modified from CogVideoXPipeline.__call__
def sample_latents_last_n_steps(
    pipeline: CogVideoXPipeline,
    latents: torch.FloatTensor,
    scheduler: Union[DDIMInverseScheduler, CogVideoXDDIMScheduler],
    last_n_steps: int,
    prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 6,
    use_dynamic_cfg: bool = False,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    reference_latents: torch.FloatTensor = None,
) -> torch.FloatTensor:
    pipeline._guidance_scale = guidance_scale
    pipeline._attention_kwargs = attention_kwargs
    pipeline._interrupt = False

    device = pipeline._execution_device

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Encode input prompt
    prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
        prompt,
        negative_prompt,
        do_classifier_free_guidance,
        device=device,
    )
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    if reference_latents is not None:
        prompt_embeds = torch.cat([prompt_embeds] * 2, dim=0)

    # 4. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(scheduler, num_inference_steps, device)
    pipeline._num_timesteps = len(timesteps)

    # 5. Prepare latents.
    latents = latents.to(device=device) * scheduler.init_noise_sigma

    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = pipeline.prepare_extra_step_kwargs(generator, eta)
    if isinstance(
        scheduler, DDIMInverseScheduler
    ):  # Inverse scheduler does not accept extra kwargs
        extra_step_kwargs = {}

    # 7. Create rotary embeds if required
    image_rotary_emb = (
        pipeline._prepare_rotary_positional_embeddings(
            height=latents.size(3) * pipeline.vae_scale_factor_spatial,
            width=latents.size(4) * pipeline.vae_scale_factor_spatial,
            num_frames=latents.size(1),
            device=device,
        )
        if pipeline.transformer.config.use_rotary_positional_embeddings
        else None
    )

    # 8. Denoising loop
    num_warmup_steps = max(len(timesteps) - num_inference_steps * scheduler.order, 0)
    skip_n_steps = len(timesteps) - last_n_steps

    with pipeline.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if pipeline.interrupt:
                continue
            if i < skip_n_steps:
                progress_bar.update()
                continue
                

            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            if reference_latents is not None:
                reference = reference_latents[i]
                reference = torch.cat([reference] * 2) if do_classifier_free_guidance else reference
                latent_model_input = torch.cat([latent_model_input, reference], dim=0)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latent_model_input.shape[0])

            # predict noise model_output
            noise_pred = pipeline.transformer(
                hidden_states=latent_model_input,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep,
                image_rotary_emb=image_rotary_emb,
                attention_kwargs=attention_kwargs,
                return_dict=False,
            )[0]
            noise_pred = noise_pred.float()

            if reference_latents is not None:  # Recover the original batch size
                noise_pred, _ = noise_pred.chunk(2)

            # perform guidance
            if use_dynamic_cfg:
                pipeline._guidance_scale = 1 + guidance_scale * (
                    (
                        1
                        - math.cos(
                            math.pi
                            * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0
                        )
                    )
                    / 2
                )
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + pipeline.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # compute the noisy sample x_t-1 -> x_t
            latents = scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs, return_dict=False
            )[0]
            latents = latents.to(prompt_embeds.dtype)

            if i == len(timesteps) - 1 or (
                (i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0
            ):
                progress_bar.update()

    # Offload all models
    pipeline.maybe_free_model_hooks()

    return latents

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
        "--video_path", type=str, required=True, help="Path of the video for inversion"
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
        "--t_0", type=float, default=0.0, help="Noise level in (0, 1)"
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
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = torch.device(args.device)
    t_0 = np.clip(args.t_0, 0.0, 1.0)
    # Set seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.makedirs(args.output_path, exist_ok=True)

    with open(args.prompt_path, "r") as f:
        prompt_dict = json.load(f)
    prompt = prompt_dict["composited_scene_caption"]
    prompt_save_path = os.path.join(args.output_path, "prompt.txt")
    with open(prompt_save_path, "w") as f:
        f.write(prompt)

    pipeline = CogVideoXPipeline.from_pretrained(
        args.model_path, torch_dtype=dtype
    ).to(device=device)

    # Get video frames
    video_frames = get_video_frames(
        video_path=args.video_path,
        width=args.width,
        height=args.height,
        skip_frames_start=0,
        skip_frames_end=0,
        max_num_frames=81,
        frame_sample_step=None,
    ).to(device=device)

    # Encode video frames
    video_latents = encode_video_frames(vae=pipeline.vae, video_frames=video_frames)
    
    # Get timesteps and alphas
    scheduler = pipeline.scheduler    
    timesteps, num_inference_steps = retrieve_timesteps(scheduler, args.num_inference_steps, device)
    alphas = pipeline.scheduler.alphas_cumprod[timesteps.cpu()]
    
    # Get alphas at t_0
    last_n_steps = int(t_0 * num_inference_steps)
    alpha_t_0 = alphas[-last_n_steps].to(device)
    # print("id_t_0", id_t_0)
    # print("alpha_t_0", alpha_t_0)

    # Add noise
    noise = torch.randn_like(video_latents)
    noisy_video_latents = video_latents * alpha_t_0.sqrt() + noise * (1 - alpha_t_0).sqrt()
    video_path = os.path.join(args.output_path, "noisy.mp4")
    export_latents_to_video(pipeline, noisy_video_latents, video_path, args.fps)

    # Denoise
    denoised_latents = sample_latents_last_n_steps(
        pipeline=pipeline,
        latents=noisy_video_latents,
        # latents=torch.randn_like(noisy_video_latents),
        scheduler=pipeline.scheduler,
        last_n_steps=last_n_steps,
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=torch.Generator(device=device).manual_seed(seed),
        reference_latents=None,
    )

    video_path = os.path.join(args.output_path, "video.mp4")
    export_latents_to_video(pipeline, denoised_latents, video_path, args.fps)


def test():
    from matplotlib import pyplot as plt

    num_inference_steps = 10
    model_path = "THUDM/CogVideoX-5B"
    dtype = torch.bfloat16
    device = torch.device("cuda")
    pipeline = CogVideoXPipeline.from_pretrained(
        model_path, torch_dtype=dtype
    ).to(device=device)
    scheduler = pipeline.scheduler    
    
    timesteps_0, num_inference_steps = retrieve_timesteps(scheduler, num_inference_steps, device)
    timesteps_0 = timesteps_0.cpu()
    print("time_step_0 shape: ", timesteps_0.shape)
    # print("timesteps_0: ", timesteps_0)

    timesteps_1 = scheduler.timesteps.cpu()
    print("time_step_1 shape: ", timesteps_1.shape)
    # print("timesteps_1: ", timesteps_1)

    if torch.equal(timesteps_0, timesteps_1):
        print("timesteps_0 and timesteps_1 are identical.")
    else:
        print("timesteps_0 and timesteps_1 are NOT identical.")

    alphas = pipeline.scheduler.alphas_cumprod[timesteps_0]
    print("alphas shape: ", alphas.shape)

    plt.plot(timesteps_0.cpu().numpy(), alphas.cpu().numpy())
    plt.savefig(f"/work/hdd/benk/cl121/CogVideo/inference/outputs/baseline_sdedit/alphas_{num_inference_steps}.png")


if __name__ == "__main__":
    main()
    # test()