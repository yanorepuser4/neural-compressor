#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#

#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional
import time
import numpy as np
import datasets
import tensorflow as tf
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from collections import defaultdict

import transformers
from transformers import (
    TF_MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    TFAutoModelForCausalLM,
    TFTrainingArguments,
    set_seed,
)
from transformers.utils.versions import require_version


logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r benchmarks/language_modeling/tensorflow/gpt_j/requirements.txt")
MODEL_CONFIG_CLASSES = list(TF_MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to use.
    """

    model_name_or_path: Optional[str] = field(
        default="EleutherAI/gpt-j-6B",
        metadata={
            "help": (
                "The model checkpoint for GPT-J weights."
            )
        },
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    precision: Optional[str] = field(
        default="fp32",
        metadata={"help": "The precision that we want to run with."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for evaluation.
    """

    dataset_name: Optional[str] = field(
        default="EleutherAI/lambada_openai", metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    
def shape_list(tensor):
        """
        Deal with dynamic shape in tensorflow cleanly.

        Args:
            tensor (`tf.Tensor` or `np.ndarray`): The tensor we want the shape of.

        Returns:
            `List[int]`: The shape of the tensor as a list.
        """
        if isinstance(tensor, np.ndarray):
            return list(tensor.shape)

        dynamic = tf.shape(tensor)

        if tensor.shape == tf.TensorShape(None):
            return dynamic

        static = tensor.shape.as_list()

        return [dynamic[i] if s is None else s for i, s in enumerate(static)]
    
    
class Inference():
    def __init__(self) -> None:
        self.dur = []
        self.infer = None
        self.batch_size = 1
        self.max_new_tokens = 1
        self.num_beams = 4
        self.tokens_to_generate = 1
        self.input_tokens = 32

    def prepare_attention_mask_for_generation(
        self,
        inputs: tf.Tensor,
        pad_token_id=50256,
        eos_token_id=50256,
    ) -> tf.Tensor:
        is_input_ids = len(inputs.shape) == 2 and inputs.dtype in (tf.int32, tf.int64)
        is_pad_token_in_inputs = (pad_token_id is not None) and tf.math.reduce_any(inputs == pad_token_id)
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (pad_token_id != eos_token_id)

        # Check if input is input_ids and padded -> only then is attention_mask defined
        if is_input_ids and is_pad_token_in_inputs and is_pad_token_not_equal_to_eos_token_id:
            return tf.cast(tf.math.not_equal(inputs, pad_token_id), dtype=tf.int32)
        else:
            return tf.ones(inputs.shape[:2], dtype=tf.int32)
        
        
    def greedy_search_cond_fn(self, generated, finished_sequences, cur_len, model_kwargs):
        """state termination condition fn."""
        return ~tf.reduce_all(finished_sequences)

        # define condition fn
    def greedy_search_body_fn(self, generated, finished_sequences, cur_len, model_kwargs):
        """state update fn."""
        print("generated: ", generated.shape)
        
        if model_kwargs.get("past_key_values") is None:
            input_ids = generated[:, :cur_len]
        else:
            input_ids = tf.expand_dims(generated[:, cur_len - 1], -1)
        

        model_inputs = {'input_ids': input_ids, 'attention_mask': model_kwargs['attention_mask']}
        #model_inputs = self.prepare_inputs_for_generation(input_ids, use_cache=True, **model_kwargs)

        start = time.time()
        
        model_outputs = self.infer(**model_inputs)
        end = time.time()

        self.dur += [end - start]
        
        if 'logits' in model_outputs.keys():
            next_token_logits = model_outputs['logits'][:, -1]
        else:
            next_token_logits = model_outputs['output_0'][:, -1]

        # pre-process distribution
        next_tokens_scores = next_token_logits

        # argmax
        next_tokens = tf.argmax(next_tokens_scores, axis=-1, output_type=tf.int32)
        

        pad_token_id = 50256
        eos_token_id = [50256]

        unfinished_seq = 1 - tf.cast(finished_sequences, tf.int32)
        next_tokens = next_tokens * unfinished_seq + pad_token_id * (1 - unfinished_seq)
        next_token_is_eos = tf.math.reduce_any(
            tf.equal(
                tf.broadcast_to(next_tokens, (len(eos_token_id), self.batch_size)), tf.expand_dims(eos_token_id, -1)
            ),
            axis=0,
        )
        finished_sequences = finished_sequences | next_token_is_eos
        
        # update `generated` and `cur_len`
        update_indices = tf.stack([tf.range(self.batch_size), tf.broadcast_to(cur_len, [self.batch_size])], axis=-1)
        
        generated = tf.tensor_scatter_nd_update(tensor=generated, indices=update_indices, updates=next_tokens)
       
        #tf.concat([input_ids, next_tokens], axis=-1)
        
        cur_len += 1
        
        #model_kwargs = self._update_model_kwargs_for_generation(model_outputs, model_kwargs)
        
        #update attention_masks
        
        attention_mask = model_kwargs["attention_mask"]
        model_kwargs["attention_mask"] = tf.concat(
                    [attention_mask, tf.ones((attention_mask.shape[0], 1), dtype=tf.int32)], axis=-1
                )
        
        #model_kwargs = self._update_model_kwargs_for_generation(
                    #model_outputs, model_kwargs, is_encoder_decoder=False
                #)
        
        
        
        return generated, finished_sequences, cur_len, model_kwargs
    
    def generate(self, data):
        input_ids = tf.convert_to_tensor([data[:-1]], dtype=tf.int32)
        pad_token_id = 50256
        cur_len = len(data)-1
        input_ids_padding = tf.ones((self.batch_size, 1), dtype=tf.int32) * (pad_token_id or 0)
        generated = tf.concat([input_ids, input_ids_padding], axis=-1)
        model_kwargs = {'attention_mask': self.prepare_attention_mask_for_generation(input_ids)}
        finished_sequences = tf.convert_to_tensor([False])
        # 1st generation step has to be run before to initialize `past_key_values`
        generated, finished_sequences, cur_len, model_kwargs = self.greedy_search_body_fn(
            generated, finished_sequences, cur_len, model_kwargs
        )

        # 2-to-n generation steps can then be run in autoregressive fashion
        # only in case 1st generation step does NOT yield EOS token though
        maximum_iterations = 0
        generated, _, cur_len, _ = tf.while_loop(
            self.greedy_search_cond_fn,
            self.greedy_search_body_fn,
            (generated, finished_sequences, cur_len, model_kwargs),
            maximum_iterations=maximum_iterations,
        )

        return generated
    
    def evaluate(self, path, tf_eval_dataset):
        parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TFTrainingArguments))
        model_args, data_args, run_args = parser.parse_args_into_dataclasses()
        model = tf.saved_model.load(path)
        self.infer = model.signatures["serving_first_iteration"]
        iteration = 50
        correct = 0
        for idx, data in enumerate(tf_eval_dataset):
            print('Running Iteration: ', idx)
            print(data)
            predictions = self.generate(data)
            
            if data[-1] == predictions[0][-1].numpy():
                correct+=1
                print("prediciton is correct")
            else:
                print("prediction is incorrrect")
            print('Time taken: ', self.dur)
            #latency_list.append(self.dur)
            if iteration and idx >= iteration:
                break
        print(correct/iteration)
        


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TFTrainingArguments))
    model_args, data_args, run_args = parser.parse_args_into_dataclasses()

    logger.setLevel(logging.INFO)
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()
    
    if run_args.seed is not None:
        set_seed(run_args.seed)

    
    raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.checkpoint,
            use_auth_token=None,
        )
        
    
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    column_names = raw_datasets["test"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    mydata = tokenizer(raw_datasets["test"][text_column_name], return_tensors="np").input_ids
    with run_args.strategy.scope():
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        
        inference = Inference()
        inference.evaluate('./converted_gpt-j-6B-2-signatures-first-second-iter', mydata)
        
        
if __name__ == "__main__":
    main()
