Step-by-Step
============

This document serves as a guidence of quantizing GPT-J with Intel® Neural Compressor.


# Prerequisite

## 1. Environment

### Installation
```shell
# Build Intel® Neural Compressor from source for this branch
git clone https://github.com/intel/neural-compressor.git
cd neural_compressor
git checkout zehao/saved_model_demo
pip install -r requirements.txt 
python setup.py install
```

### Install Requirements
```shell
pip install -r requirements.txt
```
> Note: Validated TensorFlow [Version](/docs/source/installation_guide.md#validated-software-environment).

## 2. Prepare Pretrained model

The pretrained model is provided by Huggingface. Please run the following code to save the gpt-j-6B to ```saved_model``` format: 
 ```
python prepare_model.py
 ```
The actually saved_model should be saved to ./gpt-j-6B/saved_model/1.

Therefore, just use the following code to replace the gpt-j-6B folder.
 ```
mv ./gpt-j-6B/saved_model/1 ./
rm -r ./gpt-j-6B
mv ./1 ./gpt-j-6B
 ```

## 3. Prepare Dataset
The dataset will be automatically downloaded.

# Run

## 1. Modify Configs(Optional)
The default configs for quantization is shown below:
  ```python
  # whether to apply quantization by following the rules of ITEX
  self.itex_mode = False
  # whether to apply aggressive quantization without concerning the accuracy
  self.performance_only = False
  ```
Please set them to ```True``` in the ```__init__``` function of the ```ConvertSavedModel``` class in ```convert.py``` if needed.

## 2. Quantization
  ```shell
  python quantize.py --output_dir=./output
  ```
The quantized model will be saved to './converted_gpt-j-6B'.

When running this script, two pb files will be dumped: ```extracted_graph_def.pb``` and ```converted_graph_def.pb```. 

They represent the graph_def before and after inserting qdq.

Because there is a huge number of nodes in this LLM. It's recommended to convert the ```.pb``` file to ```.pbtxt``` file to inspect the whole graph.

  ```shell
  python pb_to_pbtxt.py
  ```

The ```converted_graph_def.pb``` will be used as the default file for conversion.
Using IDE(such as VScode) to open the ```converted_graph_def.pbtxt```. It's easy to find calibrated qdq nodes before quatzable Op such as ```MatMul``` and ```ConcatV2```.

  ```
node {
  name: "StatefulPartitionedCall/transformer/h_._0/mlp/fc_in/Tensordot/MatMul_eightbit_max_StatefulPartitionedCall/transformer/h_._0/attn/q_proj/Tensordot/Reshape/frozen_max_only"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 16.84615707397461
      }
    }
  }
}

node {
  name: "StatefulPartitionedCall/transformer/h_._0/mlp/fc_in/Tensordot/MatMul_eightbit_quantize_StatefulPartitionedCall/transformer/h_._0/attn/q_proj/Tensordot/Reshape"
  op: "QuantizeV2"
  input: "StatefulPartitionedCall/transformer/h_._0/attn/q_proj/Tensordot/Reshape"
  input: "StatefulPartitionedCall/transformer/h_._0/mlp/fc_in/Tensordot/MatMul_eightbit_min_StatefulPartitionedCall/transformer/h_._0/attn/q_proj/Tensordot/Reshape/frozen_min_only"
  input: "StatefulPartitionedCall/transformer/h_._0/mlp/fc_in/Tensordot/MatMul_eightbit_max_StatefulPartitionedCall/transformer/h_._0/attn/q_proj/Tensordot/Reshape/frozen_max_only"
  attr {
    key: "T"
    value {
      type: DT_QUINT8
    }
  }
  ......
}

node {
  name: "StatefulPartitionedCall/transformer/h_._0/attn/q_proj/Tensordot/MatMul_dequantize"
  op: "Dequantize"
  input: "StatefulPartitionedCall/transformer/h_._0/mlp/fc_in/Tensordot/MatMul_eightbit_quantize_StatefulPartitionedCall/transformer/h_._0/attn/q_proj/Tensordot/Reshape"
  input: "StatefulPartitionedCall/transformer/h_._0/mlp/fc_in/Tensordot/MatMul_eightbit_quantize_StatefulPartitionedCall/transformer/h_._0/attn/q_proj/Tensordot/Reshape:1"
  input: "StatefulPartitionedCall/transformer/h_._0/mlp/fc_in/Tensordot/MatMul_eightbit_quantize_StatefulPartitionedCall/transformer/h_._0/attn/q_proj/Tensordot/Reshape:2"
  ......
}

node {
  name: "StatefulPartitionedCall/transformer/h_._0/attn/q_proj/Tensordot/MatMul"
  op: "MatMul"
  input: "StatefulPartitionedCall/transformer/h_._0/attn/q_proj/Tensordot/MatMul_dequantize"
  input: "StatefulPartitionedCall/transformer/h_._0/attn/q_proj/Tensordot/ReadVariableOp__dequant"
  ......
}
  ```

## 3. Run Benchmark 
  ```shell
  python run_benchmark.py --output_dir=./output
  ```

The following results are expected to be shown:
```shell
---------------------------------------------------------
The infrence results of original gpt-j with TF2.x API
Batch size = 1
Accuracy: 78.218%
Latency: 1195.945 ms
Throughput: 0.836 samples/sec
---------------------------------------------------------
The infrence results of converted gpt-j with TF2.x API
Batch size = 1
Accuracy: 77.228%
Latency: 1404.469 ms
Throughput: 0.712 samples/sec
```

## 4. Dump Graph(Optional)
We can also dump graph from the saved_model to check if the conversion is successful:

  ```shell
  python dump_graph_from_saved_model.py --input_model=./converted_gpt-j-6B
  ```

The dumped graph will be saved at './dumped_graph.pb'