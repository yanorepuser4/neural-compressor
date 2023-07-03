# pip install -q transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import os
import sys

sys.path.insert(0, './')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["HF_HOME"] = "/dataset/huggingface"
os.environ['TRANSFORMERS_OFFLINE'] = '0'
checkpoint = "/home/sdp/wenhuach/starcoder"
device = "cpu"  # for GPU usage or "cpu" for CPU usage
calib_dataset = load_dataset("mbpp", split="test")

tokenizer = AutoTokenizer.from_pretrained(checkpoint)


# def tokenize_function(examples):
#     example = tokenizer(examples['code'], return_tensors='pt',truncation=True)
#     return example
#
#
# calib_dataset = calib_dataset.map(tokenize_function, batched=True)


def collate_batch(batch):
    assert (len(batch) == 1)
    tmp = batch[0]['code']
    tmp = tokenizer(tmp, return_tensors='pt')['input_ids']
    return tmp


calib_dataloader = DataLoader(
    calib_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=collate_batch,
)

from neural_compressor import PostTrainingQuantConfig
from neural_compressor import quantization

recipes = {}

recipes = {"smooth_quant": True, "smooth_quant_args": {'alpha': 0.8}}
op_type_dict = {}
# if args.kl:
#     op_type_dict = {'linear': {'activation': {'algorithm': ['kl']}}}
# if args.fallback_add:
#     op_type_dict["add"] = {"weight": {"dtype": ["fp32"]}, "activation": {"dtype": ["fp32"]}}

model = AutoModelForCausalLM.from_pretrained(checkpoint,
                                             torchscript=True  ##FIXME
                                             )
for input in calib_dataloader:
    model(input)
conf = PostTrainingQuantConfig(quant_level=1, backend='ipex', excluded_precisions=["bf16"],  ##use basic tuning
                               recipes=recipes,
                               op_type_dict=op_type_dict)

q_model = quantization.fit(model,
                           conf,
                           calib_dataloader=calib_dataloader,
                           )
save_model_name = "starcoder"
q_model.save(f"{save_model_name}")
