set -x


model_path='facebook/opt-125m'
precisions=(
    'fp8_e5m2'
    'fp8_e4m3'
    'fp8_e3m4'
)
for prec in ${precisions[*]}
do
python run_clm_fp8.py --model ${model_path} \
                        --calib_iters 128 \
                        --tasks lambada_openai \
                        --batch_size 32 \
                        --alpha 0.5 \
                        --quantize \
                        --precision ${prec} \
                        --cast_e4m3 \
                        --no_lm_head
done



