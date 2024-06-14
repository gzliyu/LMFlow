#!/bin/bash
model_name_or_path="/home/nlpintern1/liyu/models/TinyLlama-1.1B-Chat-v1.0"
CUDA_VISIBLE_DEVICES=2,4,5
deepspeed --include localhost:${CUDA_VISIBLE_DEVICES} examples/evaluation_gpu.py \
--cache_dir ./.cache/ \
--arch_type topk \
--answer_type text \
--model_name_or_path ${model_name_or_path} \
--dataset_path data/wiki_en_eval \
--deepspeed examples/ds_config.json \
--inference_batch_size_per_device 1 \
--block_size 4096 \
--metric ppl

# # cpu
# python examples/evaluation_cpu.py \
# --cache_dir ./.cache/ \
# --arch_type topk \
# --answer_type text \
# --model_name_or_path ${model_name_or_path} \
# --dataset_path data/wiki_en_eval \
# --deepspeed examples/ds_config.json \
# --inference_batch_size_per_device 1 \
# --block_size 4096 \
# --use_accelerator_for_evaluator True \
# --metric ppl