# Run

## Run FP32 model
``` python
python run_llm.py --model [model_name_or_path] --to_graph [--performance]|[--accuracy --tasks lambada_openai --batch_size 8]|[--generate --max_new_tokens 10]
```

## Run BF16/FP16 model
``` python
python run_llm.py --model [model_name_or_path] --approach cast --precision [bf16|fp16]  --to_graph  [--performance]|[--accuracy --tasks lambada_openai --batch_size 8]|[--generate --max_new_tokens 10]
```

## Run FP8 model
``` python
python run_llm.py --model [model_name_or_path] --approach [dynamic|static|cast] --precision [fp8_e4m3|fp8_e5m2] --to_graph  [--performance]|[--accuracy --tasks lambada_openai --batch_size 8]|[--generate --max_new_tokens 10]
```

## Run gpt-neo 2.7b model on 4 Gaudi2 cards with FP8 E4M3 static quantization
``` python
## need use latest deepspeed-fork repo master branch
deepspeed --num_gpus=4 test_gpt_neo.py --model EleutherAI/gpt-neo-2.7B --approach static --precision fp8_e4m3 --tasks lambada_openai --batch_size 16 --accuracy --quantize
```

## Run Llama2 70b model on 8 Gaudi2 cards
``` python
## need use latest deepspeed-fork repo master branch
TOKENIZERS_PARALLELISM=true deepspeed --num_gpus=8 test_llama2.py --model /git_lfs/data/pytorch/llama2/Llama-2-70b-hf/ --approach static --precision fp8_e4m3 --tasks lambada_openai --batch_size 16 --accuracy --quantize
```
