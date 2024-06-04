#!/bin/bash
model_name_or_path="/home/nlpintern1/liyu/models/Llama-7b-hf"
CUDA_VISIBLE_DEVICES=4,5
deepspeed --include localhost:${CUDA_VISIBLE_DEVICES} examples/evaluation.py \
--arch_type decoder_only \
--answer_type text \
--model_name_or_path ${model_name_or_path} \
--dataset_path data/wiki_en_eval \
--deepspeed examples/ds_config.json \
--inference_batch_size_per_device 1 \
--block_size 4096 \
--metric ppl