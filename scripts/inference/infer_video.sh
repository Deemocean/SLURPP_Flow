#!/usr/bin/env bash
cd ./slurpp

export BASE_CKPT_DIR="$(pwd)/../models"

# Input / output video
INPUT_VIDEO=../test_data/video/GX010514_reformatted.mp4
OUTPUT_VIDEO=../outputs/inference/video/restored_ema.mp4

# Run Model Directory (same checkpoints as infer_real.sh)
JOBNAME=slurpp
CKPT_NAME="diffusion"
RUN_DIR=../models/${JOBNAME}
MG_CONFIG=${RUN_DIR}/config.yaml

CKPT=${RUN_DIR}/checkpoint/${CKPT_NAME}
STAGE2=../models/slurpp/checkpoint/cld/cld_clear.pth

# --temporal_alpha 1.0 = no smoothing; lower (e.g. 0.6) reduces flicker at the
# cost of motion ghosting. --max_frames N limits work for a quick test.
uv run infer_video.py \
    --config $MG_CONFIG \
    --checkpoint $CKPT \
    --stage2_checkpoint $STAGE2 \
    --input_video $INPUT_VIDEO \
    --output_video $OUTPUT_VIDEO \
    --temporal_alpha 0.6
