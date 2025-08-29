#!/bin/sh
#
#SBATCH --job-name=bikers_dino
#SBATCH --output=/work/hdd/benk/cl121/CogVideo/inference/outputs/logs/%j.out
#SBATCH --error=/work/hdd/benk/cl121/CogVideo/inference/outputs/logs/%j.err
#
#SBATCH --account=bfaf-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --time=2-0:00
#SBATCH --mem=64GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1

cd /work/hdd/benk/cl121/CogVideo/inference

# bash scripts/video_effects/horse_knight.sh
# bash scripts/video_effects/beagle_dragon.sh
bash scripts/video_effects/bikers_dino.sh
# bash scripts/video_effects/box_puppy.sh
# bash scripts/video_effects/coral_shark.sh
# bash scripts/video_effects/coral_jellyfish.sh
# bash scripts/video_effects/creek_deer.sh
# bash scripts/video_effects/cycle_view_explosion.sh
# bash scripts/video_effects/dancer_audience.sh
# bash scripts/video_effects/exotic_road_workers.sh
# bash scripts/video_effects/girl_beach_dog.sh
# bash scripts/video_effects/girl_living_room_bear.sh
# bash scripts/video_effects/girl_living_room_balloons.sh
# bash scripts/video_effects/jogging_river_golden.sh
# bash scripts/video_effects/kid_game_bear.sh
# bash scripts/video_effects/scuba_whale.sh
# bash scripts/video_effects/taxi_tsunami.sh
# bash scripts/video_effects/tree_sunlight_giraffe.sh
# bash scripts/video_effects/tree_sunlight_elephant.sh
# bash scripts/video_effects/wis_trees_dinosaur.sh