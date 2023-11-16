import argparse
import time
import json
import re
import torch
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import habana_frameworks.torch.hpex


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", nargs="?", default="facebook/opt-125m"
)
parser.add_argument(
    "--trust_remote_code", default=True,
    help="Transformers parameter: use the external repo")
parser.add_argument(
    "--revision", default=None,
    help="Transformers parameter: set the model hub commit number")
parser.add_argument("--dataset", nargs="?", default="NeelNanda/pile-10k", const="NeelNanda/pile-10k")
parser.add_argument("--output_dir", nargs="?", default="./saved_results")
parser.add_argument("--quantize", action="store_true")
parser.add_argument("--to_graph", action="store_true")
parser.add_argument("--approach", type=str, default='static', 
                    help="Select from ['dynamic', 'static']")
parser.add_argument("--precision", type=str, default='fp8_e4m3', 
                    help="Select from ['fp8_e4m3', 'fp8_e5m2']")
parser.add_argument("--accuracy", action="store_true")
parser.add_argument("--batch_size", default=1, type=int,
                    help="For accuracy measurement only.")
parser.add_argument("--pad_max_length", default=512, type=int,
                    help="Pad input ids to max length.")
parser.add_argument("--calib_iters", default=100, type=int,
                    help="calibration iters.")
parser.add_argument("--tasks", nargs='+', default=["wikitext"], type=str, \
                    choices=["winogrande", "copa", "piqa", "rte", "hellaswag", \
                    "openbookqa", "lambada_openai", "lambada_standard", "wikitext"],
                    help="tasks list for accuracy validation")
args = parser.parse_args()


if re.search("llama", args.model.lower()):
    import transformers
    from transformers import LlamaForCausalLM, LlamaTokenizer
    user_model = LlamaForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        device_map='hpu',
    )
    tokenizer = LlamaTokenizer.from_pretrained(args.model)
elif re.search("mpt-7b-chat", args.model.lower()):
    from mpt_7b.modeling_mpt import MPTForCausalLM
    user_model = MPTForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        revision=args.revision,
        device_map='hpu',
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    user_model.config.use_cache = True
elif re.search("falcon-7b-instruct", args.model.lower()):
    from falcon_7b_instruct.modelling_RW import RWForCausalLM
    user_model = RWForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        revision=args.revision,
        device_map='hpu',
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    user_model.config.use_cache = True
else:
    user_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        revision=args.revision,
        device_map='hpu',
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

# to channels last
user_model = user_model.to(memory_format=torch.channels_last)
user_model.eval()

if args.quantize:
    print("device:", next(user_model.parameters()).device)
    from neural_compressor.torch.quantization import get_fp8_e5m2_qconfig, get_fp8_e4m3_qconfig
    if args.precision == "fp8_e4m3":
        dtype = torch.float8_e4m3fn
        qconfig = get_fp8_e4m3_qconfig()
    else:
        dtype = torch.float8_e5m2
        qconfig = get_fp8_e5m2_qconfig()


    from neural_compressor.torch.quantization.fp8 import quantize_dynamic, quantize
    if args.approach == "dynamic":
        user_model = quantize_dynamic(user_model, dtype, inplace=True)
    else:
        # dataset
        from datasets import load_dataset
        calib_dataset = load_dataset(args.dataset, split="train").select(range(100))
        calib_dataset = calib_dataset.shuffle(seed=42)
        calib_data = []
        for examples in calib_dataset:
            calib_data.append(
                tokenizer(examples["text"], return_tensors="pt", max_length=128)
            )

        def calib_func(model):
            for i, calib_input in enumerate(calib_data):
                if i >= args.calib_iters:
                    break
                model(
                    input_ids=calib_input["input_ids"],
                    attention_mask=calib_input["attention_mask"],
                )

        user_model = quantize(user_model, qconfig, calib_func=calib_func, inplace=True)

    if args.to_graph:
        import habana_frameworks.torch.hpu.graphs as htgraphs
        user_model = htgraphs.wrap_in_hpu_graph(user_model)

if args.accuracy:
    from intel_extension_for_transformers.llm.evaluation.lm_eval import evaluate
    results = evaluate(
        model="hf-causal",
        model_args='pretrained='+args.model+',tokenizer='+args.model+',dtype=float32',
        user_model=user_model,
        batch_size=args.batch_size,
        tasks=args.tasks,
        device='hpu',
    )
    dumped = json.dumps(results, indent=2)
    for task_name in args.tasks:
        if task_name == "wikitext":
            print("Accuracy for %s is: %s" % (task_name, results["results"][task_name]["word_perplexity"]))
        else:
            print("Accuracy for %s is: %s" % (task_name, results["results"][task_name]["acc"]))

