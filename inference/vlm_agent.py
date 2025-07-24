import openai
from openai import OpenAI
from PIL import Image
import base64
import io
import re
import json 
import os
import decord
import argparse
import numpy as np

def read_api_key(path="/u/cl121/openai_api.txt"):
    with open(path, "r") as f:
        return f.read().strip()

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    buffered.seek(0)
    img_str = base64.b64encode(buffered.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"

def sample_key_frames(video_path, num_key_frames=8):
    """
    Reads a video and samples key frames uniformly.

    Args:
        video_path (str): Path to the video file.
        num_key_frames (int): Number of key frames to sample.

    Returns:
        List[PIL.Image]: List of sampled key frames
    """

    vr = decord.VideoReader(video_path)
    total_frames = len(vr)
    if num_key_frames > total_frames:
        indices = list(range(total_frames))
    else:
        indices = np.linspace(0, total_frames - 1, num=num_key_frames, dtype=int)
    frames = []
    for idx in indices:
        frame = vr[idx].asnumpy()
        img = Image.fromarray(frame)
        frames.append(img)
    return frames

def load_json_from_result(result):
    # Use regex to find the JSON part in the result
    match = re.search(r"```json\s*(\{.*?\}|\[.*?\])\s*```", result, re.DOTALL)
    if match:
        json_str = match.group(1)
        return json.loads(json_str)
    else:
        raise ValueError("No JSON found in the result.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vfx_prompt", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--system_prompt_path", type=str, default="prompt.txt")
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--num_key_frames", type=int, default=5)
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_tokens", type=int, default=1000)
    args = parser.parse_args()

    with open(args.system_prompt_path, "r") as f:
        system_prompt = f.read()

    key_frames = sample_key_frames(args.video_path, args.num_key_frames)
    key_frames_base64 = [image_to_base64(frame) for frame in key_frames]
    key_frames_prompt = [{"type": "image_url", "image_url": {"url": img_data}} for img_data in key_frames_base64]
    
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            *key_frames_prompt,
            {"type": "text", "text": f"Description: {args.vfx_prompt}"},
        ]}
    ]
    
    client = OpenAI(api_key=read_api_key())
    response = client.chat.completions.create(
        model=args.model,
        messages=messages,
        seed=args.seed,
        max_tokens=args.max_tokens,
    )
    result = response.choices[0].message.content
    result = load_json_from_result(result)
    print(result)

    # save the result to a json file
    with open(args.output_path, "w") as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    main()