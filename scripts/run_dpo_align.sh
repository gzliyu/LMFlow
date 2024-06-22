#!/bin/bash
# Please run this script under ${project_id} in project directory of

# Parses arguments
CUDA_VISIBLE_DEVICES=0,1
model_name_or_path=/home/nlpintern1/liyu/models/Meta-Llama-3-8B
dataset_path=/home/nlpintern1/liyu/dataset/GSM8KPair_06-21llama3_0621
output_dir=output_models/math_dpo
deepspeed_args="--include localhost:${CUDA_VISIBLE_DEVICES}"
# specify gpus/single gpu here by 
# `--include localhost:0,1` or `--include localhost:0`

while [[ $# -ge 1 ]]; do
  key="$1"
  case ${key} in
    -m|--model_name_or_path)
      model_name_or_path="$2"
      shift
      ;;
    -d|--dataset_path)
      dataset_path="$2"
      shift
      ;;
    -o|--output_lora_path)
      output_dir="$2"
      shift
      ;;
    --deepspeed_args)
      deepspeed_args="$2"
      shift
      ;;
    *)
      echo "error: unknown option \"${key}\"" 1>&2
      exit 1
  esac
  shift
done
exp_id=dpo
project_dir=$(cd "$(dirname $0)"/..; pwd)
log_dir=${project_dir}/log/${exp_id}
mkdir -p ${output_dir} ${log_dir}

deepspeed ${deepspeed_args} \
  examples/dpo_train.py \
    --model_name_or_path ${model_name_or_path} \
    --dataset_path ${dataset_path} \
    --output_dir ${output_dir} \
    --run_name math_dpo \
    --max_steps 200 \
    --learning_rate 1e-5 \
    --use_lora 1 \
    --lora_r 128 \
    --lora_alpha 256 \
    --sanity_check True \
    --save_aggregated_lora 0\
    --logging_steps 20 \
    | tee ${log_dir}/train.log \
    2> ${log_dir}/train.err
