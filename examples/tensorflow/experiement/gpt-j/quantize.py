import logging
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional
import time
import numpy as np
import datasets
import tensorflow as tf
from datasets import load_dataset

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
require_version("datasets>=1.8.0", "To fix: pip install -r examples/tensorflow/language-modeling/requirements.txt")
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

def prepare_dataset():
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

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    print("Tokenized Dataset:")
    print(tokenized_datasets["test"])
    L = []
    for x in tokenized_datasets["test"]:
        L.append(len(x['input_ids']))
    
    print("MAX = ", np.max(np.array(L)) )

    block_size = tokenizer.model_max_length
    if block_size > 1024:
        logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
        block_size = 1024

    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    eval_dataset = lm_datasets["test"]

    with run_args.strategy.scope():
        model = TFAutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, config=config)
        embeddings = model.get_input_embeddings()

        if hasattr(embeddings, "embeddings"):
            embedding_size = embeddings.embeddings.shape[0]
        else:
            embedding_size = embeddings.weight.shape[0]
        if len(tokenizer) > embedding_size:
            model.resize_token_embeddings(len(tokenizer))
        
        num_replicas = run_args.strategy.num_replicas_in_sync
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

        tf_eval_dataset = model.prepare_tf_dataset(
            eval_dataset,
            shuffle=False,
            batch_size=num_replicas * run_args.per_device_eval_batch_size,
            drop_remainder=True,
        ).with_options(options)

        return tf_eval_dataset

tf_eval_dataset = prepare_dataset()

def evaluate(path):
    model = tf.saved_model.load(path)
    infer = model.signatures["serving_default"]
    warmup = 1
    iteration = None
    latency_list = []
    iteration = 5
    results = []
    for idx, data in enumerate(tf_eval_dataset):
        inputs = data[0]
        for name in inputs:
            inputs[name] = tf.cast(inputs[name], tf.int32)
        start = time.time()
        predictions = infer(**inputs)
        end = time.time()
        results.append(predictions)
        latency_list.append(end - start)
        if iteration and idx >= iteration:
            break
    latency = np.array(latency_list[warmup:]).mean() / 1

    return latency, results

def main():    
    from convert import ConvertSavedModel
    converter = ConvertSavedModel(src='./gpt-j-6B', dst='./converted_gpt-j-6B', 
                                                quantize=True, evaluate=evaluate)
    converter()

if __name__ == "__main__":
    main()