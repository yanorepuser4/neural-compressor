python -u run_clm_no_trainer.py \
    --model "hf-internal-testing/tiny-random-GPTJForCausalLM" \
    --approach weight_only \
    --quantize \
    --sq \
    --alpha "auto"