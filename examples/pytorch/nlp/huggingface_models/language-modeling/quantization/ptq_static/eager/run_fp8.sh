set -x

model_path='facebook/opt-125m'
precisions=(
    'fp8_e5m2'
    'fp8_e4m3'
)
for prec in ${precisions[*]}
do
python run_clm_fp8.py --model ${model_path} \
                        --accuracy  --ipex --calib_iters 1 \
                        --tasks lambada_openai --sq \
                        --batch_size 32 --int8  \
                        --alpha 0.5 --quantize \
                        --precision ${prec} \
                        --cast_e4m3
done



