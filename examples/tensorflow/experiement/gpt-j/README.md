Step-by-Step
============

This document is used to record the experimental code of modifying unfrozen graph_def, which is extracted from saved_model, and saving back.


# Prerequisite

## 1. Environment

### Installation
```shell
# Install Intel® Neural Compressor
pip install neural-compressor
```

### Install Requirements
```shell
pip install -r requirements.txt
```
> Note: Validated TensorFlow [Version](/docs/source/installation_guide.md#validated-software-environment).

## 2. Prepare Pretrained model

The pretrained model is provided by [Keras Applications](https://keras.io/api/applications/). prepare the model, Run as follow: 
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
The dataset will be automatically downloaded when runing evaluation.

# Conversion
## 1. Convert without inserting qdq
This step is to verify the graph_def extracted from saved_model can be successfully saved back.
The reconstruted model will be saved to './converted_gpt-j-6B'. 
By checking the directory of this saved_model, we can see the variables folder is not empty.
  ```shell
  python convert.py
  ```
When running this script, two pb files will be dumped: ```extracted_graph_def.pb``` and ```converted_graph_def.pb```. 
They represent the graph_def before and after inserting qdq.

Because we didn't insert qdq in this step, they should be the same.

## 2. Run Benchmark 
This step is to verify the reconstructed saved_model can be inferenced without accuracy drop.
  ```shell
  python run_benchmark.py --output_dir=./output
  ```

The following results are expected to be shown:
  ```shell
  ---------------------------------------------------------
  The infrence results of original gpt-j with TF2.x API
  Batch size = 8
  Latency: 54857.059 ms
  Throughput: 0.018 images/sec
  ---------------------------------------------------------
  The infrence results of converted gpt-j with TF2.x API
  Batch size = 8
  Latency: 53673.399 ms
  Throughput: 0.019 images/sec
  ---------------------------------------------------------
  MSE of the output logits between two models = 0.0
  ```
Because it's complicated to rewrite the code of computing accuracy using TF2.x API instead of transformers API, MSE similarity is calculated for the logits outputs between the two models.
A zero MSE  represents that the output of two models are actually equal.
This result proves that the reconstructed model can still be inferenced with correct accuracy.

## 3. Convert with inserting qdq
This step is to verify the graph_def with inserted qdq can be successfully saved back.
The reconstruted model will be saved to './converted_gpt-j-6B'.
By checking the directory of this saved_model, we can see the variables folder is not empty.
  ```shell
  python convert.py --insert_qdq
  ```

When running this script, two pb files will be dumped: ```extracted_graph_def.pb``` and ```converted_graph_def.pb```. 
They represent the graph_def before and after inserting qdq.

Because there is a huge number of nodes in this LLM. It's recommended to convert the ```.pb``` file to ```.pbtxt``` file to inspect the whole graph.

  ```shell
  python pb_to_pbtxt.py
  ```

The ```converted_graph_def.pb``` will be used as the default file for conversion.
Using IDE(such as vscode) to open the ```converted_graph_def.pbtxt```. It's easy to see that qdq patterns has been successfully inserted before ```MatMul``` Op.

  ```
     node_def {
      name: "StatefulPartitionedCall/transformer/h_._0/attn/q_proj/Tensordot/MatMul_eightbit_quantize_StatefulPartitionedCall/transformer/h_._0/attn/q_proj/Tensordot/Reshape"
      op: "QuantizeV2"
      input: "StatefulPartitionedCall/transformer/h_._0/attn/q_proj/Tensordot/Reshape:output:0"
      input: "StatefulPartitionedCall/lm_head/Tensordot/MatMul_eightbit_min_StatefulPartitionedCall/lm_head/Tensordot/Reshape:output:0"
      input: "StatefulPartitionedCall/lm_head/Tensordot/MatMul_eightbit_max_StatefulPartitionedCall/lm_head/Tensordot/Reshape:output:0"
      ......
      }

    node_def {
      name: "StatefulPartitionedCall/transformer/h_._0/attn/q_proj/Tensordot/MatMul_dequantize"
      op: "Dequantize"
      input: "StatefulPartitionedCall/transformer/h_._0/attn/q_proj/Tensordot/MatMul_eightbit_quantize_StatefulPartitionedCall/transformer/h_._0/attn/q_proj/Tensordot/Reshape:output:0"
      input: "StatefulPartitionedCall/transformer/h_._0/attn/q_proj/Tensordot/MatMul_eightbit_quantize_StatefulPartitionedCall/transformer/h_._0/attn/q_proj/Tensordot/Reshape:output_min:0"
      input: "StatefulPartitionedCall/transformer/h_._0/attn/q_proj/Tensordot/MatMul_eightbit_quantize_StatefulPartitionedCall/transformer/h_._0/attn/q_proj/Tensordot/Reshape:output_max:0"
      ......
    }

    node_def {
      name: "StatefulPartitionedCall/transformer/h_._0/attn/q_proj/Tensordot/MatMul"
      op: "MatMul"
      input: "StatefulPartitionedCall/transformer/h_._0/attn/q_proj/Tensordot/MatMul_dequantize:output:0"
      input: "StatefulPartitionedCall/transformer/h_._0/attn/q_proj/Tensordot/ReadVariableOp:value:0"
      ......
    }
  ```

## 4. Run Benchmark 
This step is to verify the model inserted qdq can be inferenced. And there should be significant accuracy change becuase the calibration is not done. The min-max value of qdq is fixed to be -1 and 1.
  ```shell
  python run_benchmark.py --output_dir=./output
  ```

The following results are expected to be shown:
  ```shell
  ---------------------------------------------------------
  The infrence results of original gpt-j with TF2.x API
  Batch size = 8
  Latency: 56324.354 ms
  Throughput: 0.018 images/sec
  ---------------------------------------------------------
  The infrence results of converted gpt-j with TF2.x API
  Batch size = 8
  Latency: 48612.103 ms
  Throughput: 0.021 images/sec
  ---------------------------------------------------------
  MSE of the output logits between two models = 6.612018585205078
  ```

Because it's complicated to rewrite the code of computing accuracy using TF2.x API instead of transformers API, MSE similarity is calculated for the logits outputs between the two models.
A none-zero MSE represents that the output of two models are not equal(accuracy changed).
This proves that the qdq pattern has been successfully inserted, and the saved_model can still be inferenced.

## 5. Dump Graph
We can also dump graph from the saved_model to check if the conversion is successful:
  ```shell
  python dump_graph_from_saved_model.py --input_model=./converted_gpt-j-6B
  ```
The dumped graph will be saved at './dumped_graph.pb'