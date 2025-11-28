#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
export RAY_memory_usage_threshold=0.98

CUDA_IDS=0,1,2,3,4,5,6,7
N_GPU=8

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct

TOTAL_EPOCHES=2
GLOBAL_BATCH_SIZE=128
ROLLOUT_BATCH_SIZE=384
MINI_ROLLOUT_BATCH_SIZE=384
VAL_BATCH_SIZE=512
MAX_PROMPT_LENGTH=4096
ROLLOUT=8
clip_ratio_high=0.28

L_SAFE_STATIC=400
use_entopy_advantage_shaping=true
entropy_alpha=0.4
entropy_kappa=2.0


# EXP_NAME="qwen2_5_vl_7b__dapo_clip_high_${clip_ratio_high}__ep${TOTAL_EPOCHES}_rb${ROLLOUT_BATCH_SIZE}_gb${GLOBAL_BATCH_SIZE}_mini${MINI_ROLLOUT_BATCH_SIZE}_rollout${ROLLOUT}_from_vppo_length_limit_${L_SAFE_STATIC}_use_entopy_advantage_shaping_${use_entopy_advantage_shaping}_alpha_${entropy_alpha}_kappa_${entropy_kappa}"


EXP_NAME="qwen2_5_vl_7b__dapo_clip_high_${clip_ratio_high}__ep${TOTAL_EPOCHES}_rb${ROLLOUT_BATCH_SIZE}_gb${GLOBAL_BATCH_SIZE}_mini${MINI_ROLLOUT_BATCH_SIZE}_rollout${ROLLOUT}_from_vppo"


CONGI_FILE="examples/configs/config_vppo.yaml"
TRAIN_FILE="PAPOGalaxy/PAPO_ViRL39K_train"
VAL_FILE="PAPOGalaxy/PAPO_MMK12_test"

FORMAT_PROMPT="examples/format_prompt/math_perception.jinja"
REWARD_FUNCTION="examples/reward_function/math.py:compute_score_wo_format"
# REWARD_FUNCTION="examples/reward_function/math.py:compute_score_wo_format_length_limit"


CUDA_VISIBLE_DEVICES=${CUDA_IDS} python3 -m verl.trainer.main \
    config=${CONGI_FILE} \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${VAL_FILE} \
    data.rollout_batch_size=${ROLLOUT_BATCH_SIZE} \
    data.mini_rollout_batch_size=${MINI_ROLLOUT_BATCH_SIZE} \
    data.format_prompt=${FORMAT_PROMPT} \
    worker.rollout.tensor_parallel_size=1 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.global_batch_size=${GLOBAL_BATCH_SIZE} \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=${clip_ratio_high} \
    algorithm.disable_kl=true \
    algorithm.online_filtering=true \
    algorithm.filter_key=accuracy \
    algorithm.filter_low=0.01 \
    algorithm.filter_high=0.99 \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=${N_GPU} \
    trainer.total_epochs=${TOTAL_EPOCHES} \
    worker.reward.reward_function=${REWARD_FUNCTION} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    worker.rollout.n=${ROLLOUT} \
    worker.actor.micro_batch_size_per_device_for_update=2 \
    worker.actor.micro_batch_size_per_device_for_experience=8 \
    worker.reward.reward_function_kwargs.L_SAFE_STATIC=${L_SAFE_STATIC} \
    worker.actor.use_entopy_advantage_shaping=${use_entopy_advantage_shaping} \
    worker.actor.entropy_alpha=${entropy_alpha} \
    worker.actor.entropy_kappa=${entropy_kappa} \