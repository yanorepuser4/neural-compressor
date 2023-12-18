import tensorflow as tf
import numpy as np
from typing import List, Optional, Union, Tuple
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
from transformers.utils.versions import require_version

from transformers import (
    TF_MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    TFTrainingArguments,
    set_seed,
)

from int8_benchmark import Inference

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
    

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
warmup_steps = 1
batch_size = 3
steps = 1
nrows_warmup = warmup_steps * batch_size
nrows_actual = steps * batch_size
nrows = nrows_warmup + nrows_actual


rdata = ["Once upon a time, there existed a little girl, who liked to have adventures. She wanted to go to places and meet new people, and have fun."] 
rdata = rdata * nrows
mydata = tokenizer(rdata, return_tensors="tf").input_ids

# config = AutoConfig.from_pretrained(model_args.model_name_or_path)
# tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
# column_names = raw_datasets["test"].column_names
# text_column_name = "text" if "text" in column_names else column_names[0]

# mydata = tokenizer(raw_datasets["test"][text_column_name], return_tensors="np").input_ids

# marg = {}
# stacked = np.concatenate(mydata)
# unique, counts = np.unique(stacked, return_counts=True)
# counts = counts / np.sum(counts)

# marg = dict(zip(unique, counts))
# marg = defaultdict(lambda: 0, marg)


infer = Inference(dataset = mydata)

def main():    
    with run_args.strategy.scope():
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        from convert import ConvertSavedModel
        from configs import op_wise_config, int8_sequences
        converter = ConvertSavedModel(src="./gpt-j-6B-2-signatures-first-second-iter", 
                                      dst="./converted_gpt-j-6B-2-signatures-first-second-iter", 
                                      evaluate=infer.evaluate,
                                      op_wise_config=op_wise_config,
                                      int8_sequences=int8_sequences,
                                      signature_names=["serving_first_iteration", "serving_default"])
                                    
        converter()

if __name__ == "__main__":
    main()