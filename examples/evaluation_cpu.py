#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Statistics and Machine Learning Research Group at HKUST. All rights reserved.
"""A one-line summary of the module or program, terminated by a period.

Leave one blank line.  The rest of this docstring should contain an
overall description of the module or program.  Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples.

Typical usage example:

  foo = ClassFoo()
  bar = foo.FunctionBar()
"""
import json
import os
import sys
import torch
# sys.path.remove(os.path.abspath(os.path.dirname(sys.argv[0])))
from transformers import HfArgumentParser

from lmflow.datasets.dataset import Dataset
from lmflow.pipeline.auto_pipeline import AutoPipeline

from lmflow.args import ModelArguments, DatasetArguments, AutoArguments

from lmflow.models.hf_decoder_model import *
from transformers import AutoModelForCausalLM
from lmflow.models.modeling_topkllama import TopKLlamaForCausalLM
from lmflow.pipeline.evaluator import *
import multiprocessing
MODEL_MAP = {
    "topk": TopKLlamaForCausalLM,
    "auto": AutoModelForCausalLM
}



class CPUDecoderModel(DecoderModel, Tunable):
    def __init__(
        self,
        model_args,
        tune_strategy='normal',
        ds_config=None,
        use_accelerator=False,
        model_implementation="auto",
        *args,
        **kwargs
    ):
        self.device = "cpu"
        self.model_args = model_args
        tokenizer_kwargs = {
            "cache_dir": model_args.cache_dir,
            "use_fast": model_args.use_fast_tokenizer,
            "revision": model_args.model_revision,
            "use_auth_token": True if model_args.use_auth_token else None,
            "trust_remote_code": model_args.trust_remote_code,
            "model_max_length": model_args.model_max_length,
        }
        assert model_args.model_name_or_path, (
            "Missing model_name_or_path"
            "You are instantiating a new tokenizer from scratch."
            "This is not supported by this script."
        )

        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
                    
        self.tokenizer = tokenizer  

        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        logger.debug(f"torch_dtype on init: {torch_dtype}")

        config_kwargs = {
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "use_auth_token": True if model_args.use_auth_token else None,
            "trust_remote_code": model_args.trust_remote_code,
        }
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)

        #position interpolation
        if model_args.do_rope_scaling:
            if "LlamaForCausalLM" in config.architectures:
                from lmflow.utils.position_interpolation.llama_rope_scaled_monkey_patch import (
                        replace_llama_with_condense,
                )
                replace_llama_with_condense(model_args.rope_pi_ratio, model_args.rope_ntk_ratio)

        if model_args.model_max_length:
            config.max_position_embeddings = model_args.model_max_length
            config.model_max_length = model_args.model_max_length
        
        if tune_strategy == 'normal':
            compute_dtype = torch_dtype
            if model_args.use_qlora:
                model_args.use_lora = True
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=model_args.bits == 4,
                    load_in_8bit=model_args.bits == 8,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=model_args.double_quant,
                    bnb_4bit_quant_type=model_args.quant_type,
                )

            model = MODEL_MAP[model_implementation].from_pretrained(
                model_args.model_name_or_path,
                config=config,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                quantization_config=quant_config if model_args.use_qlora else None,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                torch_dtype=torch_dtype,
                trust_remote_code = model_args.trust_remote_code,
            )
            if model_args.use_qlora:
                model.gradient_checkpointing_enable()
                model = prepare_model_for_kbit_training(model)

            self.backend_model_full = model
            if model_args.use_lora:
                if model_args.lora_target_modules:
                    lora_target_modules = model_args.lora_target_modules
                else:
                    lora_target_modules = None
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=model_args.lora_r,
                    lora_alpha=model_args.lora_alpha,
                    lora_dropout=model_args.lora_dropout,
                    target_modules=lora_target_modules,
                )
                model = get_peft_model(model, peft_config)
                model.print_trainable_parameters()

            # We resize the embeddings only when necessary to avoid index errors.
            # If you are creating a model from scratch on a small vocab and want a
            # smaller embedding size, remove this test.
            if len(tokenizer) > embedding_size:
                model.resize_token_embeddings(len(tokenizer))

            self.config = config
            self.backend_model = model
            self.tune_strategy = tune_strategy
        elif tune_strategy == 'none':
            if use_accelerator:
                peft_model_id = model_args.lora_model_path
                self.backend_model = MODEL_MAP[model_implementation].from_pretrained(
                        model_args.model_name_or_path,
                        config=config,
                        device_map="cpu",
                        torch_dtype=torch_dtype,
                        load_in_8bit = model_args.use_int8,
                    )
                if peft_model_id is not None:
                    self.backend_model = PeftModel.from_pretrained(
                        self.backend_model, 
                        peft_model_id,
                    )
                self.tokenizer.padding_side = "left"
        elif tune_strategy == 'adapter':
            raise NotImplementedError('adapter tune strategy not implemented')

        if self.tokenizer.eos_token_id is None:
            self.tokenizer.eos_token_id = self.backend_model.config.eos_token_id
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def tokenize(self, dataset, add_special_tokens=True, *args, **kwargs):
        """
        Tokenize the full dataset.
    
        Parameters
        ------------
        dataset : lmflow.datasets.Dataset.

        args : Optional.
            Positional arguments.
        
        kwargs : Optional.
            Keyword arguments.    
        
        Returns
        ------------
        tokenized_datasets :
            The tokenized dataset, without any leading or trailing special
            tokens (normally they are Begin-Of-Sentence or End-Of-Sentence
            tokens).
        """
        # Preprocessing the datasets.
        # First we tokenize all the texts.
        if dataset.get_backend() != "huggingface":
            raise NotImplementedError(
                "tokenization of datasets with non-huggingface backend are"
                "not supported yet"
            )

        dataset_type = dataset.get_type()
        model_args = self.model_args
        raw_datasets = dataset
        hf_raw_datasets = dataset.get_backend_dataset()
        column_names = list(hf_raw_datasets.features)
        data_args = raw_datasets.get_data_args()

        # Requires three types of information for tokenizing different datasets
        #   1) Which fields require tokenization, e.g.
        #        "text2float": "text", but not "float"
        #        "text2text": both "input" and "output"
        #   2) How will there tokenized sequence concatenated together, e.g.
        #        "text_only": "text" -> "text"
        #        "text2text": "input", "output" -> "input" + "output"
        #   3) Which fields require loss in final computation, e.g.
        #        "text_only": "text"
        #        "text2text": "output" only
        tokenized_column_order = None       # Handles 1) and 2)
        label_columns = None                # Handles 3)
        if dataset_type == "text_only":
            tokenized_column_order = ["text"]
            label_columns = ["text"]
        elif dataset_type == "text2text":
            tokenized_column_order = ["input", "output"]
            label_columns = ["output"]
            add_special_tokens = False
        elif dataset_type == "conversation":
            if data_args.conversation_template:
                if data_args.conversation_template in PRESET_TEMPLATES.keys():
                    conversation_template = PRESET_TEMPLATES[data_args.conversation_template]
                else:
                    raise NotImplementedError(
                        f"Conversation template {data_args.conversation_template} is not supported yet."
                    )
            else:
                logger.warning("No conversation template provided. Using default template.")
                conversation_template = PRESET_TEMPLATES['empty']
                        
            logger.warning(f"Conversation template: {conversation_template}")
        else:
            raise NotImplementedError(
                f"dataset type \"{dataset_type}\" is not supported, currently"
                " only support following data types:\n"
                f"    1) {TEXT_ONLY_DATASET_DESCRIPTION}\n"
                f"    2) {TEXT2TEXT_DATASET_DESCRIPTION}\n"
                f"    3) {CONVERSATION_DATASET_DESCRIPTION}\n"
            )

        # Whether to truncate long sequences to fit into max_length
        use_truncation = False
        if model_args.use_lora or data_args.disable_group_texts:
            use_truncation = True
        
        tokenize_fn = conversation_tokenize_function if "conversation" in dataset_type else tokenize_function
        tokenize_fn_kwargs = {
            "data_args": data_args,
            "tokenizer": self.tokenizer,
            "column_names": column_names,
        }
        if "conversation" in dataset_type:
            tokenize_fn_kwargs["conversation_template"] = conversation_template
        else:
            tokenize_fn_kwargs["label_columns"] = label_columns
            tokenize_fn_kwargs["tokenized_column_order"] = tokenized_column_order
            tokenize_fn_kwargs["add_special_tokens"] = add_special_tokens
            tokenize_fn_kwargs["use_truncation"] = use_truncation
                           
        tokenize_kwargs = {}
        if not data_args.streaming:
            fingerprint = hashlib.md5(
                (
                    raw_datasets.get_fingerprint()
                    + str(self.tokenizer)
                    + ('###conversation_template=' + str(conversation_template) if "conversation" in dataset_type else "")
                    + f'###disable_group_texts={data_args.disable_group_texts}'
                    + f'###block_size={data_args.block_size}'
                ).encode("utf-8")
            ).hexdigest()
            tokenize_kwargs = {
                "num_proc": data_args.preprocessing_num_workers,
                "load_from_cache_file": not data_args.overwrite_cache,
                "desc": "Running tokenizer on dataset",
                "new_fingerprint": fingerprint,
            }

        tokenized_datasets = raw_datasets.map(
            tokenize_fn,
            batched=True,
            remove_columns=column_names,
            fn_kwargs=tokenize_fn_kwargs,
            **tokenize_kwargs
        )

        return tokenized_datasets


    def encode(self, input: Union[str, List[str]], *args, **kwargs ) -> Union[List[int], List[List[int]]]:
        """
        Perform encoding process of the tokenizer.
    
        Parameters
        ------------
        inputs : str or list.
            The text sequence.
            
        args : Optional.
            Positional arguments.
        
        kwargs : Optional.
            Keyword arguments.    
        
        Returns
        ------------
        outputs :
            if string input,return the tokenized inputs.
            "Hello,world!"-> [101, 7592, 1010, 2088, 102]
            if batch input,return {input_ids,attention_mask,token_type_ids}
            ["Hello,world!","Hello!"]-> {'input_ids': tensor([[  101,  7592,  1010,  2088,   102],...),'attention_mask': tensor([[1, 1, 1, 1, 1],[0,0,1,1,1]])}
        """
        if isinstance(input, list):
            return self.tokenizer(text=input, *args, **kwargs)#batch encode,will automatically do left padding
        elif isinstance(input, str):
            return self.tokenizer.encode(text=input, *args, **kwargs)
        else:
            raise NotImplementedError(f'type "{type(input)}" cannot be encoded')


    def decode(self, input, *args, **kwargs ) -> Union[str, List[str]]:
        """
        Perform decoding process of the tokenizer.
    
        Parameters
        ------------
        inputs : list or tensor.
            The token sequence.
            
        args : Optional.
            Positional arguments.
        
        kwargs : Optional.
            Keyword arguments.    
        
        Returns
        ------------
        outputs :
            The text decoded from the token inputs.
            if batch input,return the list of text
            [[101, 7592, 1010, 2088, 102],[101, 7592, 1010, 2088, 102]]-> ["Hello,world!","Hello,world!"
            if single input,return the text
            [101, 7592, 1010, 2088, 102]-> "Hello,world!"
        """
        if isinstance(input, List):
            input=torch.tensor(input)
        if input.dim()==2:
            return self.tokenizer.batch_decode(input, *args, **kwargs)#batch_decode
        else:
            # Can be list of ints or a Tensor
            return self.tokenizer.decode(input, *args, **kwargs)


    def inference(self, inputs, use_accelerator=False, *args, **kwargs):
        """
        Perform generation process of the model.
    
        Parameters
        ------------
        inputs :
            The sequence used as a prompt for the generation or as model inputs to the model.
            
        args : Optional.
            Positional arguments.
        
        kwargs : Optional.
            Keyword arguments.    
        
        Returns
        ------------
        outputs :
            The generated sequence output 
        """


        with torch.no_grad():
            if use_accelerator:
                outputs = self.backend_model.generate(
                    input_ids=inputs,
                    pad_token_id=self.tokenizer.pad_token_id,
                    *args,
                    **kwargs
                )
            else:
                if self.device == "gpu":
                    outputs = self.ds_engine.module.generate(
                        input_ids=inputs,
                        synced_gpus=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        *args,
                        **kwargs
                    )
                elif self.device == "cpu":
                    outputs = self.backend_model.generate(
                        input_ids=inputs,
                        synced_gpus=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        *args,
                        **kwargs
                    )
                else:
                    raise NotImplementedError(
                        f"device \"{self.device}\" is not supported"
                    )
        return outputs


    def merge_lora_weights(self):
        if self.model_args.use_lora and not self.model_args.use_qlora:
            self.get_backend_model().merge_and_unload()
        elif self.model_args.use_qlora:
            logger.warning("Reloading base model in 16-bit precision to merge adapter weights. NOTE: Your device must have"
                           "sufficient memory to reload the model in half-precision without quantization.")
            self.get_peft_without_qlora()
            self.get_backend_model().merge_and_unload()
        else:
            logger.warning("LoRA training is NOT enabled. Merging LoRA weights is not applicable.")


    def get_peft_without_qlora(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdirname:
            print('created temporary directory', tmpdirname)


            self.get_backend_model().save_pretrained(tmpdirname)

            torch_dtype = (
                self.model_args.torch_dtype
                if self.model_args.torch_dtype in ["auto", None]
                else getattr(torch, self.model_args.torch_dtype)
            )
            config_kwargs = {
                "cache_dir": self.model_args.cache_dir,
                "revision": self.model_args.model_revision,
                "use_auth_token": True if self.model_args.use_auth_token else None,
            }
            config = AutoConfig.from_pretrained(self.model_args.model_name_or_path, **config_kwargs)
            device_map = "auto"
            if os.environ.get('LOCAL_RANK') is not None:
                local_rank = int(os.environ.get('LOCAL_RANK','0'))
                device_map = {'': local_rank}

            self.backend_model_full = AutoModelForCausalLM.from_pretrained(
                self.model_args.model_name_or_path,
                from_tf=bool(".ckpt" in self.model_args.model_name_or_path),
                config=config,
                cache_dir=self.model_args.cache_dir,
                revision=self.model_args.model_revision,
                use_auth_token=True if self.model_args.use_auth_token else None,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code = self.model_args.trust_remote_code,
                attn_implementation="flash_attention_2" if self.model_args.use_flash_attention else None,
            )
        
            self.backend_model = PeftModel.from_pretrained(self.backend_model_full, tmpdirname)


    def save(self, dir, save_full_model=False, *args, **kwargs):
        """
        Perform generation process of the model.
    
        Parameters
        ------------
        dir :
            The directory to save model and tokenizer
            
        save_full_model : Optional.
            Whether to save full model.
        
        kwargs : Optional.
            Keyword arguments.    
        
        Returns
        ------------
        outputs :
            The generated sequence output 
        """
        self.get_tokenizer().save_pretrained(dir)
        if save_full_model and self.model_args.use_lora:
            save_dtype = (
                torch.float16
                if self.model_args.torch_dtype in ["auto", None]
                else getattr(torch, self.model_args.torch_dtype)
            )
            self.backend_model_full.to(dtype=save_dtype).save_pretrained(dir)
            logger.warning(f"Save full model with dtype: {save_dtype}")
        else:
            self.get_backend_model().save_pretrained(dir)


    def get_max_length(self):
        """
        Return max acceptable input length in terms of tokens.
        """
        return self.tokenizer.model_max_length


    def get_tokenizer(self):
        """
        Return the tokenizer of the model.
        """
        return self.tokenizer


    def get_backend_model(self):
        """
        Return the backend model.
        """
        return self.backend_model


    def get_attn_map(self):
        return self.backend_model_full.get_attn_map()

class CPUEvaluator(BasePipeline):
    """
    Initializes the `Evaluator` class with given arguments.

    Parameters
    ------------
    model_args : ModelArguments object.
        Contains the arguments required to load the model.
    
    data_args : DatasetArguments object.
        Contains the arguments required to load the dataset.

    evaluator_args : EvaluatorArguments object.
        Contains the arguments required to perform evaluation.


    """
    def __init__(self, model_args, data_args, evaluator_args):
    # our method
        self.data_args = data_args
        self.evaluator_args = evaluator_args
        self.model_args = model_args

        # logger
        if(self.evaluator_args.use_wandb == True):
            wandb.init(project="lmflow_evaluation")
        # random seed
        set_random_seed(self.evaluator_args.random_seed)

        self.config = AutoConfig.from_pretrained(model_args.model_name_or_path)
        try: 
            self.model_hidden_size = self.config.hidden_size
        except:
            print("Error in setting hidden size, use the default size 1024")
            self.model_hidden_size = 1024 # gpt2 seems do not have hidden_size in config

        print(f"model_hidden_size = {self.model_hidden_size}")
        # batch size has to be divisible by world_size, but can be bigger than world_size
        train_batch_size = self.evaluator_args.inference_batch_size_per_device
        self.evaluator_args.minibatch_size = train_batch_size
        self.block_size = evaluator_args.evaluate_block_size
        # dataloader, data_size = create_dataloader(args)    # load dataset


    def create_dataloader(self, dataset: Dataset):
        data_dict = dataset.to_dict()
        inputs = [ instance["input"] for instance in data_dict["instances"] ]
        outputs = [ instance["output"] for instance in data_dict["instances"] ]
        dataset_size = len(outputs)
        dataset_buf = []
        for idx in range(dataset_size):
            dataset_buf.append({
                "input": inputs[idx],
                "output": outputs[idx],
                "input_idx": idx
            })

        dataloader = batchlize(
            dataset_buf,
            self.evaluator_args.minibatch_size,
            self.evaluator_args.random_shuffle
        )
        print(f"Successfully create dataloader with size {len(dataloader)},batch_size {self.evaluator_args.minibatch_size}.")
        
        return dataloader, dataset_size


    # TODO: Split for better unittest

    def _match(self, predicted_answer, groundtruth, answer_type=None):
        case_insensitive_types = [
            "strategyqa",
            "coin_flip",
            "pubmedqa",
            "binary_choice",
            "medmcqa",
            "usmle",
        ]
        if answer_type in case_insensitive_types:
            return predicted_answer.lower() == groundtruth.lower()
        else:
            return predicted_answer == groundtruth
        return False


    def evaluate(
        self,
        model,
        dataset: Dataset,
        metric = "accuracy",
        verbose=True,
    ):
        """
        Perform Evaluation for a model

        Parameters
        ------------
        model : TunableModel object.
            TunableModel to perform inference

        dataset : Dataset object.
        """
        world_size = 16
        ppl = self._parallel_evaluate_ppl(
            model=model, 
            dataset=dataset
        )  
        print(f"Evaluating final perplexity: {ppl}")
        return ppl

    def calculate_nll_for_task(self, task):
        task_id, input_ids, target_ids = task
        print(f"Running task {task_id}.")
        with torch.no_grad():
            outputs = model.get_backend_model()(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        # 这里我们只返回一个负对数似然值，稍后将合并这些值来计算平均值
        return neg_log_likelihood

    def _parallel_evaluate_ppl(
        self,
        model, 
        dataset, 
        verbose=True
    ):
        data_dict = dataset.to_dict()
        if data_dict['type'] == 'text2text':
            raise NotImplementedError("ppl evaluation is currently not supported for text2text dataset, please use text_only dataset.")
        texts = [ instance["text"] for instance in data_dict["instances"] ]
        encodings = model.get_tokenizer()("\n\n".join(texts), return_tensors="pt")
        # Define some constant
        if self.model_args.truncate_to_model_max_length:
            max_length = model.get_max_length()
        else:
            max_length = self.block_size
        
        if verbose:
            print(f"The maximum sequence length : {max_length}")
        seq_len = encodings.input_ids.size(1)

        nlls = []
        prev_end_loc = 0
        
        tasks = []
        task_id = 0
        for begin_loc in range(0, seq_len, self.block_size):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from block_size on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc]
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100
            
            tasks.append((
                task_id,
                input_ids, 
                target_ids, 
            ))
            
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
            task_id += 1
        # with multiprocessing.Pool(processes=3) as pool:
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.map(self.calculate_nll_for_task, tasks)

        # 合并结果并计算平均NLL
        nlls = torch.stack(results)
        mean_nll = nlls.mean()
        ppl = torch.exp(mean_nll)
        return ppl
    
    
    def _evaluate_ppl(
        model, 
        dataset, 
        model_args, 
        block_size, 
        verbose=True
    ):
        data_dict = dataset.to_dict()
        if data_dict['type'] == 'text2text':
            raise NotImplementedError("ppl evaluation is currently not supported for text2text dataset, please use text_only dataset.")
        texts = [ instance["text"] for instance in data_dict["instances"] ]
        encodings = model.get_tokenizer()("\n\n".join(texts), return_tensors="pt")
        # Define some constant
        if self.model_args.truncate_to_model_max_length:
            # try:
            #     max_length = min(model.get_backend_model().config.n_positions, model.get_max_length())
            # except:
            #     max_length = min(1024, model.get_max_length())
            max_length = model.get_max_length()
        else:
            max_length = self.block_size
        
        if verbose:
            print(f"The maximum sequence length : {max_length}")
        seq_len = encodings.input_ids.size(1)

        nlls = []
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, self.block_size):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from block_size on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc]
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model.get_backend_model()(input_ids, labels=target_ids)
                # loss is calculated using CrossEntropyLoss which averages over valid labels
                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                # to the left by 1.
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)
            prev_end_loc = end_loc
            if verbose:
                print(f"Evaluating PPL: {int(begin_loc/self.block_size) + 1} / {int(seq_len/self.block_size)} Complete, current ppl : {torch.exp(torch.stack(nlls).mean())}")
            if end_loc == seq_len:
                break
        ppl = torch.exp(torch.stack(nlls).mean())
        return ppl


if __name__ == "__main__":
    pipeline_name = "evaluator"
    PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)

    parser = HfArgumentParser((ModelArguments, DatasetArguments, PipelineArguments))
    model_args, data_args, pipeline_args = parser.parse_args_into_dataclasses()

    with open (pipeline_args.deepspeed, "r") as f:
        ds_config = json.load(f)

    if data_args.block_size:
        model_args.model_max_length = data_args.block_size


    model = CPUDecoderModel(
        model_args=model_args, 
        tune_strategy='none', 
        ds_config=ds_config, 
        use_accelerator=pipeline_args.use_accelerator_for_evaluator
    )

    dataset = Dataset(data_args)

    evaluator = CPUEvaluator(
        model_args=model_args,
        data_args=data_args,
        evaluator_args=pipeline_args,
    )
    evaluator.evaluate(
        model=model, 
        dataset=dataset, 
        metric=pipeline_args.metric
    )
    torch.save(model.get_attn_map(), "attn_map.pt")