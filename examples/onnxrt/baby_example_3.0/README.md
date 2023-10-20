``` shell
# clone repo
git clone https://github.com/intel-innersource/frameworks.ai.lpot.intel-lpot.git lpot
cd lpot
git checkout -b inc3 origin/ly/inc3

# create a new conda env
conda create -n inc3_test_20 python=3.8 -y
conda activate inc3_test_20

# install requirements for neural-compressor
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:$PWD

# install requirements for demo example
cd ./examples/onnxrt/baby_example_3.0
pip install -r requirements.txt

# run demo
python ort_hello_world.py

# If everything is all right, you are expected to get following logs:
(inc3_test_20) [st_liu@tpcx-bb baby_example_3.0]$ python ort_hello_world.py
2023-09-20 10:06:12 [INFO] ****************************************
2023-09-20 10:06:12 [INFO] There are 2 tuning stages in total.
2023-09-20 10:06:12 [INFO] Stage 0: sq_alpha.
2023-09-20 10:06:12 [INFO] Stage 1: ort_graph_opt_level.
2023-09-20 10:06:12 [INFO] ****************************************
2023-09-20 10:06:12 [INFO] Start tuning stage: sq_alpha
2023-09-20 10:06:12 [INFO] Quantizing model with config: {'alpha': 0.1}
2023-09-20 10:06:12 [INFO] Evaluating model: FakeModel
2023-09-20 10:06:12 [INFO] Quantizing model with config: {'alpha': 0.2}
2023-09-20 10:06:12 [INFO] Evaluating model: FakeModel
2023-09-20 10:06:12 [INFO] Quantizing model with config: {'alpha': 0.3}
2023-09-20 10:06:12 [INFO] Evaluating model: FakeModel
2023-09-20 10:06:12 [INFO] ****************************************
2023-09-20 10:06:12 [INFO] Start tuning stage: ort_graph_opt_level
2023-09-20 10:06:12 [INFO] Quantizing model with config: {'ort_graph_opt_level': <OptimizationLevel.DISABLED: <GraphOptimizationLevel.ORT_DISABLE_ALL: 0>>}
2023-09-20 10:06:12 [INFO] Evaluating model: FakeModel
2023-09-20 10:06:12 [INFO] Quantizing model with config: {'ort_graph_opt_level': <OptimizationLevel.BASIC: <GraphOptimizationLevel.ORT_ENABLE_BASIC: 1>>}
2023-09-20 10:06:12 [INFO] Evaluating model: FakeModel
2023-09-20 10:06:12 [INFO] Quantizing model with config: {'ort_graph_opt_level': <OptimizationLevel.EXTENDED: <GraphOptimizationLevel.ORT_ENABLE_EXTENDED: 2>>}
2023-09-20 10:06:12 [INFO] Evaluating model: FakeModel
2023-09-20 10:06:12 [INFO] Quantizing model with config: {'ort_graph_opt_level': <OptimizationLevel.ALL: <GraphOptimizationLevel.ORT_ENABLE_ALL: 99>>}
2023-09-20 10:06:12 [INFO] Evaluating model: FakeModel
2023-09-20 10:06:12 [INFO] ****************************************

```