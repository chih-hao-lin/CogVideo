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
    paths_group = [
        # beagle + dragon
        [
            "/work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results_f9/beagle/beagle.mp4",
            "/work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results_f9/beagle/beagle_dragon.mp4",
            "/work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/beagle_dragon/sampling_0_seed_0/output_4.mp4",
            "/work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/beagle_dragon/sampling_0_seed_0/merged.mp4"
        ],
        # bikers + dino
        [
            "/work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results_f9/bikers/bikers.mp4",
            "/work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results_f9/bikers/bikers_dino.mp4",
            "/work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/bikers_dino/sampling_0_seed_0/output_4.mp4",
            "/work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/bikers_dino/sampling_0_seed_0/merged.mp4"
        ],
        # box + puppy
        [
            "/work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results_f9/box/box.mp4",
            "/work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results_f9/box/box_puppy.mp4",
            "/work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/box_puppy/sampling_0_seed_0/output_4.mp4",
            "/work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/box_puppy/sampling_0_seed_0/merged.mp4"
        ],
        # coral + shark
        [
            "/work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results_f9/coral/coral.mp4",
            "/work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results_f9/coral/coral_shark.mp4",
            "/work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/coral_shark/sampling_0_seed_1/output_4.mp4",
            "/work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/coral_shark/sampling_0_seed_1/merged.mp4"
        ],
        # coral + jellyfish
        [
            "/work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results_f9/coral/coral.mp4",
            "/work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results_f9/coral/cora_jellyfish.mp4",
            "/work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/coral_jellyfish/sampling_0_seed_0/output_4.mp4",
            "/work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/coral_jellyfish/sampling_0_seed_0/merged.mp4"
        ],
        # creek + deer
        [
            "/work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results_f9/creek/creek.mp4",
            "/work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results_f9/creek/creek_deer.mp4",
            "/work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/creek_deer/sampling_0_seed_0/output_4.mp4",
            "/work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/creek_deer/sampling_0_seed_0/merged.mp4"
        ],
        # cycle_view + explosion
        [
            "/work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results_f9/cycle_view/cycle_view.mp4",
            "/work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results_f9/cycle_view/cycle_view_explosion.mp4",
            "/work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/cycle_view_explosion/sampling_0_seed_1/output_4.mp4",
            "/work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/cycle_view_explosion/sampling_0_seed_1/merged.mp4"
        ],
        # dancer_stage + audience
        [
            "/work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results_f9/dancer_stage/dancer_stage.mp4",
            "/work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results_f9/dancer_stage/dancer_audience.mp4",
            "/work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/dancer_audience/sampling_0_seed_0/output_4.mp4",
            "/work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/dancer_audience/sampling_0_seed_0/merged.mp4"
        ],
        # exotic_road + workers
        [
            "/work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results_f9/exotic_road/exotic_road.mp4",
            "/work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results_f9/exotic_road/exotic_road_workers.mp4",
            "/work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/exotic_road_workers/sampling_0_seed_0/output_4.mp4",
            "/work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/exotic_road_workers/sampling_0_seed_0/merged.mp4"
        ],
        # girl_beach + dog
        [
            "/work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results_f9/girl_beach/girl_beach.mp4",
            "/work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results_f9/girl_beach/beach_dog.mp4",
            "/work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/girl_beach_dog/sampling_0_seed_0/output_4.mp4",
            "/work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/girl_beach_dog/sampling_0_seed_0/merged.mp4"
        ],
        # girl_living_room + bear
        [
            "/work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results_f9/girl_living_room/girl_living_room.mp4",
            "/work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results_f9/girl_living_room/girl_living_room_bear.mp4",
            "/work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/girl_living_room_bear/sampling_0_seed_0/output_4.mp4",
            "/work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/girl_living_room_bear/sampling_0_seed_0/merged.mp4"
        ],
        # girl_living_room + balloons
        [
            "/work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results_f9/girl_living_room/girl_living_room.mp4",
            "/work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results_f9/girl_living_room/girl_living_room_balloons.mp4",
            "/work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/girl_living_room_balloons/sampling_0_seed_0/output_4.mp4",
            "/work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/girl_living_room_balloons/sampling_0_seed_0/merged.mp4"
        ],
        # horse + knight
        [
            "/work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results_f9/horse/horse.mp4",
            "/work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results_f9/horse/horse_knight.mp4",
            "/work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/horse_knight/sampling_0_seed_0/output_4.mp4",
            "/work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/horse_knight/sampling_0_seed_0/merged.mp4"
        ],
        # jogging_river + golden
        [
            "/work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results_f9/jogging_river/jogging_river.mp4",
            "/work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results_f9/jogging_river/jogging_river_golden.mp4",
            "/work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/jogging_river_golden/sampling_0_seed_0/output_4.mp4",
            "/work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/jogging_river_golden/sampling_0_seed_0/merged.mp4"
        ],
        # kid_game + bear
        [
            "/work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results_f9/kid_game/kid_game.mp4",
            "/work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results_f9/kid_game/kid_game_bear.mp4",
            "/work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/kid_game_bear/sampling_0_seed_0/output_4.mp4",
            "/work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/kid_game_bear/sampling_0_seed_0/merged.mp4"
        ],
        # scuba + whale
        [
            "/work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results_f9/scuba/scuba.mp4",
            "/work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results_f9/scuba/scuba_whale.mp4",
            "/work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/scuba_whale/sampling_0_seed_0/output_4.mp4",
            "/work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/scuba_whale/sampling_0_seed_0/merged.mp4"
        ],
        # taxi + tsunami
        [
            "/work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results_f9/taxi/taxi.mp4",
            "/work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results_f9/taxi/taxi_tsunami.mp4",
            "/work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/taxi_tsunami/sampling_0_seed_0/output_4.mp4",
            "/work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/taxi_tsunami/sampling_0_seed_0/merged.mp4"
        ],
        # tree_sunlight + giraffe
        [
            "/work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results_f9/tree_sunlight/tree_sunlight.mp4",
            "/work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results_f9/tree_sunlight/tree_sunlight_giraffe.mp4",
            "/work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/tree_sunlight_giraffe/sampling_0_seed_0/output_4.mp4",
            "/work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/tree_sunlight_giraffe/sampling_0_seed_0/merged.mp4"
        ],
        # tree_sunlight + elephant
        [
            "/work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results_f9/tree_sunlight_2/tree_sunlight_2.mp4",
            "/work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results_f9/tree_sunlight_2/tree_sunlight_elephant.mp4",
            "/work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/tree_sunlight_elephant/sampling_0_seed_0/output_4.mp4",
            "/work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/tree_sunlight_elephant/sampling_0_seed_0/merged.mp4"
        ],
        # wis_trees + dinosaur
        [
            "/work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results_f9/wis_trees/wis_trees.mp4",
            "/work/hdd/benk/cl121/dynvfx.github.io/sm/assets/results_f9/wis_trees/wis_trees_5.mp4",
            "/work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/wis_trees_dinosaur/sampling_0_seed_0/output_4.mp4",
            "/work/hdd/benk/cl121/CogVideo/inference/outputs/dynvfx/wis_trees_dinosaur/sampling_0_seed_0/merged.mp4"
        ]
    ]

    # Process each group of videos
    for group_idx, paths in tqdm(enumerate(paths_group)):
        print(f"Processing group {group_idx + 1}/{len(paths_group)}")
            
        # Read all videos
        vrs = [decord.VideoReader(p) for p in paths[:3]]
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

        # Create output directory
        output_dir = "/work/hdd/benk/cl121/CogVideo/inference/outputs/merged_videos"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filename based on the effect name
        path_save = paths[-1]

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
        
        print(f"Saved merged video: {path_save}")
    

if __name__ == "__main__":
    merge_videos()