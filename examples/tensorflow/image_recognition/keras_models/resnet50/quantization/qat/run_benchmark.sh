#!/bin/bash
set -x

function main {
  batch_size=32
  init_params "$@"
  run_benchmark

}

# init params
function init_params {
  batch_size=32

  for var in "$@"
  do
    case $var in
      --input_model=*)
          input_model=$(echo $var |cut -f2 -d=)
      ;;
      --mode=*)
          mode=$(echo $var |cut -f2 -d=)    
      ;;
      --dataset_location=*)
          dataset_location=$(echo $var |cut -f2 -d=)    
      ;;
      --batch_size=*)
          batch_size=$(echo $var |cut -f2 -d=)    
      ;;
    esac
  done

}

# run_tuning
function run_benchmark {

    python main.py \
            --input_model ${input_model} \
            --benchmark \
            --mode ${mode} \
            --batch_size ${batch_size} \
            --dataset_location ${dataset_location}
}

main "$@"