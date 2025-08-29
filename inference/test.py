import glob
import shutil
import imageio
import decord
import os
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image

def clip_video_to_n_frames(source_video_path, target_video_path, n_frames=9, fps=8):
    """
    Clips the input video to the first n_frames and saves it to the target path.
    """
    # Use decord to read video
    vr = decord.VideoReader(source_video_path)
    total_frames = len(vr)
    # If video has fewer than n_frames, just copy it
    if total_frames <= n_frames:
        shutil.copy2(source_video_path, target_video_path)
        return
    # Otherwise, extract first n_frames
    frames = [vr[i].asnumpy() for i in range(n_frames)]
    # Save using imageio
    
    writer = imageio.get_writer(target_video_path, fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()

def process_videos_clip_n_frames(source_folder, target_folder, n_frames=9, fps=8, video_exts=(".mp4", ".avi", ".mov", ".mkv")):
    """
    Parse all videos in source_folder, clip to n_frames, and save to target_folder with same structure.
    """
    os.makedirs(target_folder, exist_ok=True)
    # Get all video files recursively
    video_files = []
    for ext in video_exts:
        video_files.extend(glob.glob(os.path.join(source_folder, "**", f"*{ext}"), recursive=True))
    for src_path in tqdm(video_files):
        # Compute relative path
        rel_path = os.path.relpath(src_path, source_folder)
        tgt_path = os.path.join(target_folder, rel_path)
        tgt_dir = os.path.dirname(tgt_path)
        os.makedirs(tgt_dir, exist_ok=True)
        clip_video_to_n_frames(src_path, tgt_path, n_frames=n_frames, fps=fps)

def clip_first_frame(source_video_path, target_img_path):
    """
    Clips the input video to the first n_frames and saves it to the target path.
    """
    # Use decord to read video
    vr = decord.VideoReader(source_video_path)
    frame = vr[0].asnumpy()
    Image.fromarray(frame).save(target_img_path)

def process_videos_clip_first_frame(source_folder, target_folder, video_exts=(".mp4", ".avi", ".mov", ".mkv")):
    """
    Parse all videos in source_folder, clip to first frame, and save to target_folder with same structure.
    """
    os.makedirs(target_folder, exist_ok=True)
    # Get all video files recursively
    video_files = []
    for ext in video_exts:
        video_files.extend(glob.glob(os.path.join(source_folder, "**", f"*{ext}"), recursive=True))
    for src_path in tqdm(video_files):
        # Compute relative path
        rel_path = os.path.relpath(src_path, source_folder)
        tgt_path = os.path.join(target_folder, rel_path.replace(".mp4", ".png"))
        tgt_dir = os.path.dirname(tgt_path)
        os.makedirs(tgt_dir, exist_ok=True)
        clip_first_frame(src_path, tgt_path)

def main():
    source_folder = "/work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results"
    target_folder = "/work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results_img"
    # process_videos_clip_n_frames(source_folder, target_folder, n_frames=12, fps=8)
    process_videos_clip_first_frame(source_folder, target_folder)

def merge_videos():
    paths = [
        "/work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results_f9/horse/horse.mp4",
        "/work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results_f9/horse/horse_knight.mp4",
        "/work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/horse_f12/test_iterative_4_seed_4/output_4.mp4"
    ]
    path_save = "/work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/horse_f12/test_iterative_4_seed_4/merged.mp4"

    # Read all videos
    vrs = [decord.VideoReader(p) for p in paths]
    # Use the minimum frame count among all videos
    min_len = min(len(vr) for vr in vrs)
    # Use the fps of the first video
    fps = vrs[0].get_avg_fps()

    # Get frame size for each video
    frame_shapes = [vr[0].asnumpy().shape for vr in vrs]
    # Resize all frames to the same height (use min height)
    min_height = min(shape[0] for shape in frame_shapes)

    # Compute new widths for each video to keep aspect ratio
    resized_widths = []
    for shape in frame_shapes:
        h, w = shape[0], shape[1]
        new_w = int(w * min_height / h)
        resized_widths.append(new_w)
    total_width = sum(resized_widths)

    # Write merged video using imageio
    with imageio.get_writer(path_save, fps=fps, codec='libx264', quality=8) as writer:
        for i in range(min_len):
            frames = []
            for idx, vr in enumerate(vrs):
                frame = vr[i].asnumpy()
                # Resize to min_height, keep aspect
                frame_resized = cv2.resize(frame, (resized_widths[idx], min_height), interpolation=cv2.INTER_AREA)
                frames.append(frame_resized)
            merged = np.concatenate(frames, axis=1)
            # imageio expects RGB
            writer.append_data(merged)
    

if __name__ == "__main__":
    merge_videos()