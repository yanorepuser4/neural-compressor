# Run

## Run FP32 model
``` python
python run_llm.py --model {model_name_or_path} --tasks lambada_openai  --[accuracy|generate] [--batch_size 4|--max_new_tokens 10]
```

## Run FP8 model
``` python
python run_llm.py --model {model_name_or_path} --approach [dynamic|static|cast] --precision [fp8_e4m3|fp8_e5m2] --tasks lambada_openai --to_graph  --[accuracy|generate] [--batch_size 4|--max_new_tokens 10]
```
