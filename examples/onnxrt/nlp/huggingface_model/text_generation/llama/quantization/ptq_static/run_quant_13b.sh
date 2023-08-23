#!/bin/bash
set -x

function main {
  init_params "$@"
  run_tuning
}

# init params
function init_params {
  for var in "$@"
  do
    case $var in
      --input_model=*)
          input_model=$(echo $var |cut -f2 -d=)
      ;;
      --output_model=*)
          output_model=$(echo $var |cut -f2 -d=)
      ;;
      --quant_format=*)
          quant_format=$(echo $var |cut -f2 -d=)
      ;;
      --batch_size=*)
          batch_size=$(echo $var |cut -f2 -d=)
      ;;
      --dataset=*)
          dataset=$(echo $var |cut -f2 -d=)
      ;;
      --smooth_quant=*)
          smooth_quant=$(echo $var |cut -f2 -d=)
      ;;
      --alpha=*)
          alpha=$(echo $var |cut -f2 -d=)
      ;;
      --tokenizer=*)
          tokenizer=$(echo $var |cut -f2 -d=)
      ;;
      --weight_only=*)
          weight_only=$(echo $var |cut -f2 -d=)
      ;;
      --algorithm=*)
          algorithm=$(echo $var |cut -f2 -d=)
      ;;
      --group_size=*)
          group_size=$(echo $var |cut -f2 -d=)
      ;;
      --scheme=*)
          scheme=$(echo $var |cut -f2 -d=)
      ;;
      --workspace=*)
	  workspace=$(echo $var |cut -f2 -d=)
      ;;
    esac
  done

}

# run_tuning
function run_tuning {
    extra_cmd=""
    # Check if the input_model ends with the filename extension ".onnx"
    if [[ $input_model =~ \.onnx$ ]]; then
        # If the string ends with the filename extension, get the path of the file
        input_model=$(dirname "$input_model")
    fi

    # Check if the output_model ends with the filename extension ".onnx"
    if [[ $output_model =~ \.onnx$ ]]; then
        # If the string ends with the filename extension, get the path of the file
        output_model=$(dirname "$output_model")
    fi

    if [[ $smooth_quant == "True" ]]; then
        # If the string ends with the filename extension, get the path of the file
        extra_cmd+=" --smooth_quant "
    fi

    if [[ $weight_only == "True" ]]; then
        extra_cmd+=" --weight_only "
    fi

    # Check if the directory exists
    if [ ! -d "$output_model" ]; then
        # If the directory doesn't exist, create it
	mkdir -p "$output_model"
	echo "Created directory $output_model"
    fi

    python llama2-13b-hf.py \
            --quant_format ${quant_format-QOperator} \
            --model_path ${input_model} \
	    --tokenizer ${tokenizer-/home/azure-node-inc/llama-2-13-hf-onnx} \
            --output_model ${output_model} \
            --batch_size ${batch_size-1} \
            --smooth_quant_alpha ${alpha-0.6} \
            --dataset ${dataset-NeelNanda/pile-10k} \
            --algorithm ${algorithm-None} \
            --group_size ${group_size-32} \
            --scheme ${scheme-sym} \
	    --workspace ${workspace-nc_worskspace} \
            --tune \
            ${extra_cmd}
}

main "$@"


