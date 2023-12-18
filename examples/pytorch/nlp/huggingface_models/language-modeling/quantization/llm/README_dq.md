Step-by-Step
============
This document describes the step-by-step instructions to run large language models (LLMs) on 4th Gen Intel® Xeon® Scalable Processor (codenamed Sapphire Rapids) with PyTorch.

The script `run_clm_no_trainer.py` supports GPT-J 6B additional INT8 quantization request.

# Prerequisite
## 1. Create Environment
```
# Installation
pip install -r requirements.txt
```

# Run

Here is how to run the scripts:

**Causal Language Modeling (CLM)**

`run_clm_no_trainer.py` quantizes the large language models using the dataset [NeelNanda/pile-10k](https://huggingface.co/datasets/NeelNanda/pile-10k) calibration and validates `lambada_openai`, `piqa`, `winogrande`, `hellaswag` and other datasets accuracy provided by lm_eval, an example command is as follows.
### GPT-J-6b

#### Quantization
```bash
# "--sq" is used to enable smooth quant
python run_clm_no_trainer.py \
    --model EleutherAI/gpt-j-6B \
    --quantize \
    --sq \
    --alpha 1.0 \
    --output_dir "saved_results" \
```
applying dynamic quantization to Embedding(Line:336) and test accuracy
```
# "--output_dir" is the sq int8 model saved dir
python run_clm_no_trainer.py \
    --model EleutherAI/gpt-j-6B \
    --int8 \
    --embdding \
    --output_dir "saved_results" \
    --accuracy
```