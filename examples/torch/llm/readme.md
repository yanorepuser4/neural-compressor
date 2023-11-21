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
