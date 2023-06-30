import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.functional import F

from torch.autograd import Function

from datasets import load_from_disk
from torch.utils.data import DataLoader

# import smooth_quant
from evaluation import evaluate as lm_evaluate
import os
from transformers import set_seed
import sys
from neural_compressor.compression.pruner.pruning import TickTockPruning

sys.path.insert(0, './')


def prune(model, dataloader, prune_cnt=40, completed_pruned_cnt=1, calib_cnt=100):
    for n, m in model.named_parameters():
        # print(n)

        m.requires_grad = True

    pruner = TickTockPruning(config, model, prune_cnt, completed_pruned_cnt)
    pruner.on_train_begin()
    model.eval()
    cnt = 0
    while 1:
        for inputs in dataloader:
            if cnt % 100 == 0:
                end_time = time.time() - start_time
                print(cnt, end_time, flush=True)
            if isinstance(inputs, dict):
                input_id = inputs["input_ids"]
            else:
                input_id = inputs[0]
            ##print(input_id.shape)

            input_id = input_id.to(device)
            output = model(input_id, labels=input_id)
            loss = output[0] / gradient_accumulation_steps
            loss.requires_grad_(True)

            # if args.layer_wise_training:
            #     loss.to("cpu")
            # model.to("cpu")
            loss.backward()
            pruner.update_score()
            for n, m in model.named_parameters():
                if m.grad != None:
                    m.grad.zero_()

            if cnt == calib_cnt:
                # pruner.update_score()
                # for n, m in model.named_parameters():
                #     if m.grad != None:
                #         m.grad.zero_()

                for n, m in model.named_parameters():
                    m.grad = None
                torch.cuda.empty_cache()
                pruner.step()
                break
            cnt += 1
        if cnt == calib_cnt:
            break
    model.eval()
    del pruner
    torch.cuda.empty_cache()
    torch.save(model, "gpt-j-6b-prune.pt")
    print("pruning acc")
    os.system("python3 eval.py --model_name /models/gpt-j-6B")


def train(model, optimizer, lr_scheduler, train_dataloader, gradient_accumulation_steps=1, training_step=1000):
    for n, m in model.named_parameters():
        # print(n)
        if "ln" in n:
            m.requires_grad = True
        else:
            m.requires_grad = False
    model.train()
    model = model.to(device)

    cnt = 1
    import time

    results = {}
    start_time = time.time()
    model.train()
    print("start training", flush=True)
    while 1:
        for inputs in train_dataloader:
            if cnt % 100 == 0:
                end_time = time.time() - start_time
                print(cnt, end_time, flush=True)
            if isinstance(inputs, dict):
                input_id = inputs["input_ids"]
            else:
                input_id = inputs[0]
            ##print(input_id.shape)

            input_id = input_id.to(device)
            output = model(input_id, labels=input_id)
            loss = output[0] / gradient_accumulation_steps
            loss.requires_grad_(True)

            # if args.layer_wise_training:
            #     loss.to("cpu")
            # model.to("cpu")
            loss.backward()
            # print(output[0])
            ##print(f"{cnt}/{len(calib_dataloader)}, {output.loss}")
            if cnt % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
            if cnt == training_step:
                break
            cnt += 1
        if cnt == training_step:
            break
    model.eval()
    torch.cuda.empty_cache()
    torch.save(model, "gpt-j-6b-prune.pt")
    print("training acc")
    os.system("python3 eval.py --model_name /models/gpt-j-6B")


# import json
if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HOME"] = "/models/huggingface"
    os.environ['TRANSFORMERS_OFFLINE'] = '0'
    set_seed(42)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", nargs="?", default="/models/gpt-j-6B"
    )
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="eval batch_size")
    parser.add_argument("--fp16", action='store_true',
                        help=" fp16 ")
    parser.add_argument("--gas", default=1, type=int,
                        help="gradient accumulate step")
    parser.add_argument("--device", default=3, type=str,
                        help="device gpu int number, or 'cpu' ")
    args = parser.parse_args()
    model_name = args.model_name
    model = AutoModelForCausalLM.from_pretrained(
        model_name, low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if args.device == "cpu":
        device_str = "cpu"
    else:
        device_str = f"cuda:{int(args.device)}"
    device = torch.device(device_str)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if "opt" in args.model_name:
        model.seqlen = model.config.max_position_embeddings
    else:
        model.seqlen = 2048


    def tokenize_function(examples):
        example = tokenizer(examples["text"], truncation=True, max_length=512)
        return example


    dataset_name = "NeelNanda/pile-10k"
    if os.path.exists(dataset_name.split('/')[-1]):
        calib_dataset = load_from_disk(dataset_name.split('/')[-1])
    else:
        calib_dataset = load_dataset(dataset_name, split="train")
        calib_dataset.save_to_disk(dataset_name.split('/')[-1])

    # calib_dataset = load_dataset(
    #     'allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    # )
    calib_dataset = calib_dataset.shuffle(seed=42)
    calib_dataset = calib_dataset.map(tokenize_function, batched=True)
    calib_dataset.set_format(type='torch', columns=['input_ids'])

    train_dataloader = DataLoader(
        calib_dataset,
        batch_size=1,
        shuffle=True,
        ##collate_fn=collate_batch,
    )
    import copy

    calib_dataloader = DataLoader(
        copy.deepcopy(calib_dataset),
        batch_size=1,
        shuffle=False,
        ##collate_fn=collate_batch,
    )
    cnt = 0
    gradient_accumulation_steps = args.gas
    import time

    start_time = time.time()

    forward_cnt = 10
    configs = [
        {  ## Example of a regular configuration

            # A list of modules that would be pruned. All linear/conv layers will be hooked when op_names is not explicitly defined.
            "start_step": forward_cnt,
            # Step at which to begin pruning, if a gradient-based criterion is used (e.g., snip-momentum), start_step should be equal to or greater than 1.
            "end_step": forward_cnt+100,  # Step at which to end pruning, for one-shot pruning start_step = end_step.
            "excluded_op_names": ['.*embeddings*', 'lm_head'],  # A list of modules that would not be pruned.
            'target_sparsity': 0.5,  # Target sparsity ratio of modules.
            "pruning_frequency": 250,
            "pruning_type": "snip_momentum",
            # Frequency of applying pruning, The recommended setting is one fortieth of the pruning steps.
            "pattern": "1x1",  # Default pruning pattern.
            "low_memory_usage": 'True',
            "pruning_scope": 'local',
        }
    ]  #
    from neural_compressor.training import prepare_compression, WeightPruningConfig

    config = WeightPruningConfig(configs)
    # dataloader.shuffle()
    for n, m in model.named_parameters():
        # print(n)
        if "ln" in n:
            m.requires_grad = True
        else:
            m.requires_grad = False
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = torch.optim.Adam(parameters, lr=1e-3, weight_decay=0)
    # optimizer = optim.SGD(), lr=0.1

    from transformers import get_scheduler

    training_step = 1000
    prune_cnt = 10
    gradient_accumulation_steps = 1
    total_training_step = ((prune_cnt - 1)+prune_cnt) *training_step
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(total_training_step * 0.05) // gradient_accumulation_steps,
        num_training_steps=total_training_step // gradient_accumulation_steps,
    )

    for i in range(1, prune_cnt + 1):
        prune(model, calib_dataloader, prune_cnt=prune_cnt, completed_pruned_cnt=i, calib_cnt=500)
        train(model, optimizer, lr_scheduler, train_dataloader, gradient_accumulation_steps, training_step)
    for i in range(1, prune_cnt):
        train(model, optimizer, lr_scheduler, train_dataloader, gradient_accumulation_steps, training_step)

    # while 1:
    #     for inputs in calib_dataloader:
    #         if cnt % 1 == 0:
    #             end_time = time.time() - start_time
    #             print(cnt, end_time, flush=True)
    #         if isinstance(inputs, dict):
    #             input_id = inputs["input_ids"]
    #         else:
    #             input_id = inputs[0]
    #         ##print(input_id.shape)
    #
    #         input_id = input_id.to(device)
    #         output = model(input_id, labels=input_id)
    #         loss = output[0] / gradient_accumulation_steps
    #         loss.requires_grad_(True)
    #
    #         # if args.layer_wise_training:
    #         #     loss.to("cpu")
    #         # model.to("cpu")
    #         loss.backward()
    #         pruner.update_score()
    #         for n,m in model.named_parameters():
    #             if m.grad!=None:
    #                 m.grad.zero_()
    #         # print(output[0])
    #         ##print(f"{cnt}/{len(calib_dataloader)}, {output.loss}")
    #         # if cnt % gradient_accumulation_steps == 0:
    #         #     optimizer.step()
    #         #     optimizer.zero_grad()
    #         #     lr_scheduler.step()
    #         if cnt == forward_cnt:
    #             for n, m in model.named_parameters():
    #                 m.grad = None
    #             torch.cuda.empty_cache()
    #             pruner.step()
    #             break
    #         cnt += 1
    #     if cnt == forward_cnt:
    #         break

    # results = lm_evaluate(model="hf-causal",
    #                       model_args=f'pretrained="{model_name}",tokenizer="{model_name}",dtype=float16',
    #                       user_model=model, tasks=["lambada_openai"],
    #                       device=device_str, batch_size=64)
