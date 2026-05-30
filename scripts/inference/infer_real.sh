#!/usr/bin/env bash
cd ./slurpp

export BASE_CKPT_DIR="$(pwd)/../models"

# Dataset and output configuration
DATA_DIR=../test_data
DATASET_NAME=test_data

OUTPUT_DIR=../outputs/inference/${DATASET_NAME}

#Run Model Directory
JOBNAME=slurpp
CKPT_NAME="diffusion"
RUN_DIR=../models/${JOBNAME}
MG_CONFIG=${RUN_DIR}/config.yaml

CKPT=${RUN_DIR}/checkpoint/${CKPT_NAME}
STAGE2=../models/slurpp/checkpoint/cld/cld_clear.pth

uv run infer_real.py --config $MG_CONFIG --checkpoint $CKPT --output_dir $OUTPUT_DIR --stage2_checkpoint $STAGE2 --data_dir $DATA_DIR
