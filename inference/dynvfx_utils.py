import argparse
import json
import math
import os
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
from ddim_inversion import (
    get_video_frames,
    encode_video_frames,
    export_latents_to_video
)

class DynVFXAttnProcessor(CogVideoXAttnProcessor2_0):
    def __init__(self):
        super().__init__()

    def calculate_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn: Attention,
        batch_size: int,
        image_seq_length: int,
        text_seq_length: int,
        attention_mask: Optional[torch.Tensor],
        image_rotary_emb: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2) # [b, attn.heads, num_tokens_q, head_dim]
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)     # [b, attn.heads, num_tokens_k, head_dim]
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2) # [b, attn.heads, num_tokens_v, head_dim]

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query[:, :, text_seq_length:] = apply_rotary_emb(
                query[:, :, text_seq_length:], image_rotary_emb
            )
            if not attn.is_cross_attention:
                if key.size(2) == query.size(2):  # Attention for reference hidden states
                    key[:, :, text_seq_length:] = apply_rotary_emb(
                        key[:, :, text_seq_length:], image_rotary_emb
                    )
                else:  # RoPE should be applied to each group of image tokens
                    key[:, :, text_seq_length : text_seq_length + image_seq_length] = (
                        apply_rotary_emb(
                            key[:, :, text_seq_length : text_seq_length + image_seq_length],
                            image_rotary_emb,
                        )
                    )
                    key[:, :, text_seq_length * 2 + image_seq_length :] = apply_rotary_emb(
                        key[:, :, text_seq_length * 2 + image_seq_length :], image_rotary_emb
                    )
        
        attn_mask = None
        if attention_mask is not None:
            b, attn_heads, lq, _ = query.shape
            lk = key.shape[2]
            # create a boolean tensor of shape [b, attn_heads, lq, lk]
            attn_mask = torch.ones(b, attn_heads, lq, lk, device=query.device, dtype=torch.bool)
            attn_mask[:, :, :, text_seq_length + image_seq_length:text_seq_length * 2 + image_seq_length] = False
            attn_mask[:, :, :, text_seq_length * 2 + image_seq_length:] = attention_mask[None, None]

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor, # [b, num_tokens=t*(h/patch_size)*(w/patch_size), embed_dim=3072]
        encoder_hidden_states: torch.Tensor, # [b, text_seq_length, embed_dim=3072]
        attention_mask: Optional[torch.Tensor] = None, # [1, num_tokens], bool
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image_seq_length = hidden_states.size(1)
        text_seq_length = encoder_hidden_states.size(1)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query, query_reference = query.chunk(2)
        key, key_reference = key.chunk(2)
        value, value_reference = value.chunk(2)
        batch_size = batch_size // 2

        hidden_states, encoder_hidden_states = self.calculate_attention(
            query=query,
            key=torch.cat((key, key_reference), dim=1),
            value=torch.cat((value, value_reference), dim=1),
            attn=attn,
            batch_size=batch_size,
            image_seq_length=image_seq_length,
            text_seq_length=text_seq_length,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )
        hidden_states_reference, encoder_hidden_states_reference = self.calculate_attention(
            query=query_reference,
            key=key_reference,
            value=value_reference,
            attn=attn,
            batch_size=batch_size,
            image_seq_length=image_seq_length,
            text_seq_length=text_seq_length,
            attention_mask=None,
            image_rotary_emb=image_rotary_emb,
        )

        return (
            torch.cat((hidden_states, hidden_states_reference)),
            torch.cat((encoder_hidden_states, encoder_hidden_states_reference)),
        )


class OverrideAttnProcessors:
    def __init__(self, transformer: CogVideoXTransformer3DModel):
        self.transformer = transformer
        self.original_processors = {}

    def __enter__(self):
        for block in self.transformer.transformer_blocks:
            block = cast(CogVideoXBlock, block)
            self.original_processors[id(block)] = block.attn1.get_processor()
            block.attn1.set_processor(DynVFXAttnProcessor())

    def __exit__(self, _0, _1, _2):
        for block in self.transformer.transformer_blocks:
            block = cast(CogVideoXBlock, block)
            block.attn1.set_processor(self.original_processors[id(block)])

def get_latents_mask(
    mask_frames: torch.Tensor,
    pipeline: CogVideoXPipeline,
):
    mask_ratio = torch.sum(mask_frames > 0) / mask_frames.numel()
    mask_latents = encode_video_frames(vae=pipeline.vae, video_frames=mask_frames)
    # [1, t, d, h, w] e.g., [1, 3, 16, 60, 90]

    black_frames = torch.ones_like(mask_frames) * (-1)
    black_latents = encode_video_frames(vae=pipeline.vae, video_frames=black_frames)

    latents_diff = mask_latents - black_latents # [b, t, d, h, w ]
    diff_norm = latents_diff.norm(dim=2).to(torch.float32)
    threshold = torch.quantile(diff_norm, 1 - mask_ratio)
    mask = diff_norm >= threshold
    return mask # [b, t, h, w]

def get_tokens_mask(
    mask_frames: torch.Tensor,
    pipeline: CogVideoXPipeline,
    encoder_hidden_states: torch.Tensor,
):
    text_seq_length = encoder_hidden_states.shape[1]
    mask_ratio = torch.sum(mask_frames > 0) / mask_frames.numel()
    mask_latents = encode_video_frames(vae=pipeline.vae, video_frames=mask_frames)
    mask_tokens = pipeline.transformer.patch_embed(encoder_hidden_states, mask_latents)
    mask_tokens = mask_tokens[:, text_seq_length:] # [b, num_tokens, embed_dim]

    black_frames = torch.ones_like(mask_frames) * (-1)
    black_latents = encode_video_frames(vae=pipeline.vae, video_frames=black_frames)
    black_tokens = pipeline.transformer.patch_embed(encoder_hidden_states, black_latents)
    black_tokens = black_tokens[:, text_seq_length:] # [b, num_tokens, embed_dim]

    diff_tokens = mask_tokens - black_tokens
    diff_norm = diff_tokens.norm(dim=-1).to(torch.float32)
    threshold = torch.quantile(diff_norm, 1 - mask_ratio)
    mask = diff_norm >= threshold

    return mask # [b, num_tokens]

def apply_anchor_extended_attention_dropout(
    tokens_mask: torch.Tensor, 
    dropout_fg: float = 1.0,
    dropout_bg: float = 0.0,
):
    """
    Applies dropout for Anchor Extended Attention by randomly flipping mask values
    with probability drop_ratio.
    """
    rand = torch.rand(*tokens_mask.shape, device=tokens_mask.device)
    mask_fg = (rand < dropout_fg) & tokens_mask
    mask_bg = (rand < dropout_bg) & ~tokens_mask
    tokens_mask = mask_fg | mask_bg
    return tokens_mask # [b, num_tokens]

# Modified from CogVideoXPipeline.__call__
def sample_anchor_extended_attention(
    pipeline: CogVideoXPipeline,
    latents: torch.FloatTensor,
    mask_frames: torch.FloatTensor,
    scheduler: Union[DDIMInverseScheduler, CogVideoXDDIMScheduler],
    prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_inference_steps: int = 50,
    last_n_steps: int = 50,
    guidance_scale: float = 6,
    use_dynamic_cfg: bool = False,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    reference_latents: torch.FloatTensor = None,
    aea_dropout_fg: float = 1.0,
    aea_dropout_bg: float = 0.0,
) -> torch.FloatTensor:
    pipeline._guidance_scale = guidance_scale
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
    # prompt_embeds.shape: [2, text_seq_length, 4096]

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

    # 8. Get mask
    tokens_mask = get_tokens_mask(
        mask_frames=mask_frames,
        pipeline=pipeline,
        encoder_hidden_states=prompt_embeds[0:1],
    )
    # apply Anchor Extended Attention Dropout
    tokens_mask_aea = apply_anchor_extended_attention_dropout(
        tokens_mask, dropout_fg=aea_dropout_fg, dropout_bg=aea_dropout_bg
    )
    attention_kwargs = {
        "attention_mask": tokens_mask_aea,
    }
    pipeline._attention_kwargs = attention_kwargs

    # 9. Denoising loop
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
def test_override_attn_processors():
    """
    Unit test function to test OverrideAttnProcessors using sample_latents_last_n_steps.
    
    This test verifies that:
    1. OverrideAttnProcessors correctly replaces attention processors
    2. The custom DynVFXAttnProcessor works with sample_latents_last_n_steps
    3. Attention processors are properly restored after context exit
    4. The sampling produces different results with custom vs original processors
    """
    import random
    import numpy as np
    
    # Test configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    seed = 42
    
    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    
    print(f"Testing OverrideAttnProcessors on device: {device}")
    print(f"Using dtype: {dtype}")
    
    # Load a small model for testing (you may need to adjust the model path)
    model_path = "THUDM/CogVideoX-5B"
    print(f"Loading model from: {model_path}")
        
    pipeline = CogVideoXPipeline.from_pretrained(
        model_path, torch_dtype=dtype
    ).to(device=device)
        
    # Create test data
    batch_size = 1
    num_frames = 3
    height = 480  # Small size for faster testing
    width = 720
    latent_height = height // 8
    latent_width = width // 8
        
    # Create test latents
    test_latents = torch.randn(
        batch_size, num_frames, 16, latent_height, latent_width,
        device=device, dtype=dtype
    )
    print("test_latents.shape:", test_latents.shape)
        
    # Create test prompt
    test_prompt = "A beautiful sunset over the ocean"
        
    # Test parameters
    num_inference_steps = 10  # Small number for faster testing
    last_n_steps = 5
    guidance_scale = 6.0
        
    # # Test 1: Baseline with original processors
    # print("Running baseline test with original attention processors...")

    # baseline_result = sample_latents_last_n_steps(
    #     pipeline=pipeline,
    #     latents=test_latents.clone(),
    #     scheduler=pipeline.scheduler,
    #     prompt=test_prompt,
    #     num_inference_steps=num_inference_steps,
    #     last_n_steps=last_n_steps,
    #     guidance_scale=guidance_scale,
    #     generator=torch.Generator(device=device).manual_seed(seed),
    # )

    embed_dim = pipeline.transformer.patch_embed.embed_dim
    patch_size = pipeline.transformer.patch_embed.patch_size
    patch_size_t = pipeline.transformer.patch_embed.patch_size_t
    use_positional_embeddings = pipeline.transformer.patch_embed.use_positional_embeddings
    use_learned_positional_embeddings = pipeline.transformer.patch_embed.use_learned_positional_embeddings
    print("embed_dim: ", embed_dim)
    print("patch_size: ", patch_size)
    print("patch_size_t: ", patch_size_t)
    print("use_positional_embeddings: ", use_positional_embeddings)
    print("use_learned_positional_embeddings: ", use_learned_positional_embeddings)

    # Test 2: With OverrideAttnProcessors
    print("Running test with OverrideAttnProcessors...")
    with OverrideAttnProcessors(pipeline.transformer):
        # Verify that processors have been replaced
        for block in pipeline.transformer.transformer_blocks:
            block = cast(CogVideoXBlock, block)
            processor = block.attn1.get_processor()
            assert isinstance(processor, DynVFXAttnProcessor), \
                f"Expected DynVFXAttnProcessor, got {type(processor)}"
        
        print("✓ Attention processors successfully replaced")
        
        # attention_kwargs = {
        #     "attention_mask": torch.zeros_like(test_latents)
        # }
            
        # Test sampling with custom processors
        custom_result = sample_latents_last_n_steps(
            pipeline=pipeline,
            latents=test_latents.clone(),
            scheduler=pipeline.scheduler,
            prompt=test_prompt,
            num_inference_steps=num_inference_steps,
            last_n_steps=last_n_steps,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device=device).manual_seed(seed),
            # attention_kwargs=attention_kwargs,
        )
        
@torch.no_grad()
def test_mask():
    from PIL import Image
    import numpy as np
    device = torch.device("cuda")
    dtype = torch.bfloat16
    model_path = "THUDM/CogVideoX-5B"
    video_path = "/work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/horse_f12/mask_orig.mp4"
    
    pipeline = CogVideoXPipeline.from_pretrained(
        model_path, torch_dtype=dtype
    ).to(device=device)

    mask_frames = get_video_frames(
        video_path=video_path,
        width=720,
        height=480,
        skip_frames_start=0,
        skip_frames_end=0,
        max_num_frames=81,
        frame_sample_step=None,
    ).to(device=device)

    # get latents mask
    # mask = get_latents_mask(mask_frames, pipeline)
    # print("mask.shape:", mask.shape)
    # print("mask.dtype:", mask.dtype)
    # mask_ratio = torch.sum(mask) / mask.numel()
    # print("mask_ratio:", mask_ratio)

    # mask = mask.squeeze(0)
    # for i in range(mask.shape[0]):
    #     mask_i = mask[i].cpu().numpy()
    #     mask_i = (mask_i * 255).astype(np.uint8)
    #     mask_i = Image.fromarray(mask_i)
    #     save_path = os.path.join("/work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/horse_f12/test_mask", f"mask_{i}.png")
    #     mask_i.save(save_path)

    # get tokens mask
    prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
        prompt="A beautiful sunset over the ocean",
        negative_prompt="",
        do_classifier_free_guidance=False,
        device=device,
    )
    print("prompt_embeds.shape:", prompt_embeds.shape)
    encoder_hidden_states = prompt_embeds
    mask = get_tokens_mask(
        mask_frames,
        pipeline, 
        encoder_hidden_states
    )
    mask = apply_anchor_extended_attention_dropout(mask, dropout_fg=1.0, dropout_bg=0.0)
    print("mask.shape:", mask.shape)
    print("mask.dtype:", mask.dtype)
    mask_ratio = torch.sum(mask) / mask.numel()
    print("mask_ratio:", mask_ratio)

@torch.no_grad()
def test_sample_anchor_extended_attention():
    """
    Unit test function to test sample_anchor_extended_attention.
    SDEdit + Anchor Extended Attention
    """
    from dynvfx import set_seed
    parser = argparse.ArgumentParser()
    parser.add_argument("--t_0", type=float, default=0.8)
    parser.add_argument("--aea_dropout_fg", type=float, default=1.0)
    parser.add_argument("--aea_dropout_bg", type=float, default=0.0)
    args = parser.parse_args()
    t_0 = args.t_0
    aea_dropout_fg = args.aea_dropout_fg
    aea_dropout_bg = args.aea_dropout_bg
    device = torch.device("cuda")
    dtype = torch.bfloat16
    model_path = "THUDM/CogVideoX-5B"
    video_path = "/work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results_f12/horse/horse.mp4"
    mask_path = "/work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/horse_f12/mask_orig.mp4"
    prompt_path = "/work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/horse_f12/vlm_agent.json"
    latents_path = "/work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/horse_f12/inversion.pt"
    dir_save = f"/work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/test_sample_aea/outloop_t0_{t_0}_aea_fg_{aea_dropout_fg}_bg_{aea_dropout_bg}"
    os.makedirs(dir_save, exist_ok=True)
    height, width = 480, 720
    num_frames = 81
    num_inference_steps = 250
    seed = 42
    set_seed(seed)

    with open(prompt_path, "r") as f:
        prompt_dict = json.load(f)
    prompt = prompt_dict["composited_scene_caption"]

    pipeline = CogVideoXPipeline.from_pretrained(
        model_path, torch_dtype=dtype
    ).to(device=device)
    
    video_frames = get_video_frames(
        video_path=video_path,
        width=width,
        height=height,
        skip_frames_start=0,
        skip_frames_end=0,
        max_num_frames=num_frames,
        frame_sample_step=None,
    ).to(device=device)

    # Encode video frames
    video_latents = encode_video_frames(vae=pipeline.vae, video_frames=video_frames)

    mask_frames = get_video_frames(
        video_path=mask_path,
        width=width,
        height=height,
        skip_frames_start=0,
        skip_frames_end=0,
        max_num_frames=num_frames,
        frame_sample_step=None,
    ).to(device=device)

    # Get timesteps and alphas
    scheduler = pipeline.scheduler    
    timesteps, num_inference_steps = retrieve_timesteps(scheduler, num_inference_steps, device)
    alphas = pipeline.scheduler.alphas_cumprod[timesteps.cpu()]
    
    # Get alphas at t_0
    last_n_steps = int(t_0 * num_inference_steps)
    alpha_t_0 = alphas[-last_n_steps].to(device)

    # Add noise
    noise = torch.randn_like(video_latents)
    noisy_video_latents = video_latents * alpha_t_0.sqrt() + noise * (1 - alpha_t_0).sqrt()
    path_noisy = os.path.join(dir_save, "noisy.mp4")
    export_latents_to_video(pipeline, noisy_video_latents, path_noisy, fps=8)

    inverse_latents = torch.load(latents_path).to(device)
    # Denoise
    with OverrideAttnProcessors(pipeline.transformer):
        denoised_latents = sample_anchor_extended_attention(
            pipeline=pipeline,
            latents=noisy_video_latents,
            mask_frames=mask_frames,
            scheduler=pipeline.scheduler,
            last_n_steps=last_n_steps,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=6.0,
            generator=torch.Generator(device=device).manual_seed(seed),
            reference_latents=reversed(inverse_latents),
            aea_dropout_fg=aea_dropout_fg,
            aea_dropout_bg=aea_dropout_bg,
        )

    path_denoised = os.path.join(dir_save, "denoised.mp4")
    export_latents_to_video(pipeline, denoised_latents, path_denoised, fps=8)

def test_indexing():
    tensor = torch.ones(2, 1000, 1, 2, 1)
    rand = torch.rand(2, 1000)
    threshold = 0.5
    mask = rand < threshold
    print("mask.sum():", mask.sum())

    tensor_masked = tensor[mask]
    print("tensor_masked.shape:", tensor_masked.shape)
    print("tensor_masked.sum():", tensor_masked.sum())

if __name__ == "__main__":
    # test_override_attn_processors()
    # test_mask()
    # test_indexing()
    test_sample_anchor_extended_attention()
    