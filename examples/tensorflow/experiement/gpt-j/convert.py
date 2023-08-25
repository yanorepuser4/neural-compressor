import os
import numpy as np
import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.training import saver
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.python.grappler import tf_optimizer
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.eager import context
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.training import saver
from tensorflow.core.framework import variable_pb2
from tensorflow.python.eager import wrap_function
from tensorflow.python.util import nest
from tensorflow.python.saved_model import save
from utils import parse_saved_model, reconstruct_saved_model
# from insert_qdq import GenerateGraphWithQDQPattern
from configs import op_wise_config, int8_sequences

class ConvertSavedModel():
    def __init__(self, src='./gpt-j-6B', dst='./converted_gpt-j-6B', 
                        evaluate=None, op_wise_config={}, int8_sequences={}):
        self.src = src
        self.dst = dst
        self.fp32_ops = []
        self.bf16_ops = []
        self.new_api = True
        self.device ='cpu'
        self.itex_mode = False
        self.fake_quant = False 
        self.evaluate = evaluate
        self.performance_only = False
        self.op_wise_config = op_wise_config
        self.int8_sequences = int8_sequences
        self.weight_tensor_minmax_dict = {}

    def inc_preoptimize(self, graph_def):
        from neural_compressor import Model
        from neural_compressor.adaptor.tf_utils.graph_rewriter.generic.pre_optimize import PreOptimization
        pre_optimizer_handle = PreOptimization(Model(graph_def), self.new_api, self.device)
        pre_optimized_model = pre_optimizer_handle.get_optimized_model(self.itex_mode)
        return pre_optimized_model.graph_def

    def _search_y_pattern_for_itex(self, graph_def):
        """Search the Y pattern for itex and return the op name."""
        from neural_compressor.adaptor.tf_utils.graph_util import GraphAnalyzer
        g = GraphAnalyzer()
        g.graph = graph_def
        g.parse_graph()
        y_pattern = [['Conv2D', 'MatMul'], ['BiasAdd'], ['Add', 'AddV2', 'AddN'], ('Relu',)]
        y_pattern_variant = [['MaxPool', 'AvgPool'], ['Add', 'AddV2', 'AddN'], ('Relu',)]
        target_nodes = g.query_fusion_pattern_nodes(y_pattern)
        target_nodes_variant = g.query_fusion_pattern_nodes(y_pattern_variant)

        res = {}
        for i in target_nodes:
            if i[2] not in res:
                res[i[2]] = 1
            else:
                res[i[2]] += 1
        matched_add_nodes = [(i,) for i in res if res[i] == 2]
        for i in res:
            if res[i] == 1:
                for j in target_nodes_variant:
                    if j[1] == i:
                        matched_add_nodes.append((i,))
        return matched_add_nodes

    def _adjust_weight(self, graph_def):
        """In-place adjust weight by scale.

        Args:
            scale: smooth scale with the shape (ic,)
            weight_node: reference to the original const weight node
            original_weight: numpy value of the original const weight node
        """
        # scale: (ic,)
        from utils import weight_name_mapping
        reconstruct_saved_model(graph_def, self.func, self.frozen_func, self._saved_model, self.dst)
        model = load.load(self.dst, [tag_constants.SERVING])
        for idx, weight_tensor in enumerate(model.variables):
            parsed_weight_name = weight_name_mapping(weight_tensor.name)
            if parsed_weight_name in self.sq_weight_scale_dict:
                W = np.transpose(weight_tensor, [1, 0])
                W *= self.sq_weight_scale_dict[parsed_weight_name]
                W = np.transpose(W, [1, 0])
                tf.compat.v1.assign(model.variables[idx], W)
                if parsed_weight_name not in self.weight_tensor_minmax_dict:
                    self.weight_tensor_minmax_dict[parsed_weight_name] = [np.min(W), np.max(W)]
        return model

    def _inference(self, sampling_graph_def):
        import time
        print('Inference the saved_model and capture outputs to files')
        model = self._adjust_weight(sampling_graph_def)
        start = time.time()
        _, _ = self.evaluate(model, iter=1)
        end = time.time()
        print('Calibration Inference Time: ', end-start)

    def quantize(self, graph_def):
        import copy
        import tempfile
        from neural_compressor.utils.utility import CaptureOutputToFile
        from neural_compressor.adaptor.tf_utils.graph_util import GraphRewriterHelper as Helper
        from neural_compressor.adaptor.tf_utils.graph_rewriter.qdq.insert_qdq_pattern import GenerateGraphWithQDQPattern
        from neural_compressor.adaptor.tf_utils.quantize_graph.qdq.optimize_qdq import OptimizeQDQGraph
        from neural_compressor.adaptor.tf_utils.graph_rewriter.int8.freeze_value import FreezeValueTransformer
        from neural_compressor.adaptor.tf_utils.graph_rewriter.generic.insert_print_node import InsertPrintMinMaxNode
        from neural_compressor.adaptor.tf_utils.graph_rewriter.qdq.merge_duplicated_qdq import MergeDuplicatedQDQOptimizer
        from neural_compressor.adaptor.tf_utils.graph_rewriter.generic.strip_unused_nodes import StripUnusedNodesOptimizer
        from neural_compressor.adaptor.tf_utils.graph_rewriter.qdq.share_qdq_y_pattern import ShareQDQForItexYPatternOptimizer
        from neural_compressor.adaptor.tf_utils.graph_rewriter.generic.fuse_pad_with_fp32_conv import FusePadWithFP32Conv2DOptimizer

        self.quantized_node_info = OptimizeQDQGraph(graph_def,
                                        ['attention_mask', 'input_ids'],
                                        ['Identity', 'Identity_1'],
                                        self.op_wise_config,
                                        self.int8_sequences,
                                        self.device,
                                        self.fake_quant,
                                        self.new_api,
                                        self.performance_only,
                                        self.itex_mode).get_quantized_nodes()

        if self.itex_mode:
            self.quantized_node_info.extend(self._search_y_pattern_for_itex(graph_def))

        if not self.quantized_node_info:
            return graph_def

        print('Start to do calibration')
        # Calibration using sampling model
        sampling_graph_def = copy.deepcopy(graph_def)
        # TODO: this is a workaround to make Min/Max node be completly eliminated in int8 graph
        # after enabling pad+conv2d in new API.
        non_pad_ops = list(list(set(self.fp32_ops).union(set(self.bf16_ops))))
        sampling_graph_def = FusePadWithFP32Conv2DOptimizer(
                                    sampling_graph_def,
                                    non_pad_ops,
                                    ['attention_mask', 'input_ids'],
                                    self.op_wise_config,
                                    self.new_api,
                                    True).do_transformation()

        for i in self.quantized_node_info:
            sampling_graph_def, _ = InsertPrintMinMaxNode(
                sampling_graph_def, i[0], i[-1], self.new_api).do_transformation()

        tmp_dump_file = tempfile.mkstemp(suffix='.log')[1]

        with CaptureOutputToFile(tmp_dump_file):
            self._inference(sampling_graph_def)
        self._calibration_data = Helper.gen_valid_sampling_log(tmp_dump_file)

        del sampling_graph_def
        import gc
        gc.collect()

        # Insert QDQ pattern
        self._tmp_graph_def = GenerateGraphWithQDQPattern(
              graph_def, self._calibration_data, self.op_wise_config, self.fake_quant, 
              self.fp32_ops, self.bf16_ops, self.quantized_node_info, self.device, 
              self.performance_only, self.itex_mode, self.weight_tensor_minmax_dict).do_transformation()
        
        self._tmp_graph_def, _ = FreezeValueTransformer(
            self._tmp_graph_def,
            self._calibration_data,
            '__max:',
            self.itex_mode).do_transformation()
        self._tmp_graph_def, _ = FreezeValueTransformer(
            self._tmp_graph_def,
            self._calibration_data,
            '__min:',
            self.itex_mode).do_transformation()
        self._tmp_graph_def, _= FreezeValueTransformer(
            self._tmp_graph_def,
            self._calibration_data,
            '__requant_min_max',
            tensor_data= {},
            device=self.device,
            itex_mode=self.itex_mode).do_transformation()

        self._tmp_graph_def = StripUnusedNodesOptimizer(
            self._tmp_graph_def,
            ['attention_mask', 'input_ids'],
            ['Identity', 'Identity_1']).do_transformation()

        if self.itex_mode:
            self._tmp_graph_def = ShareQDQForItexYPatternOptimizer(self._tmp_graph_def).do_transformation()
        self._tmp_graph_def = MergeDuplicatedQDQOptimizer(self._tmp_graph_def).do_transformation()

        return self._tmp_graph_def

    def smooth_quant(self, model_path, calib_iter=1, tune_cfg=None, alpha=0.491, folding=False,
                     percentile=99.999, op_types=['MatMul', 'Conv2D'], scales_per_op=True):
        """Convert the model by smooth quant.

        Args:
            model: original model
            dataloader: the calibration dataloader
            calib_iter: how many steps of iterations on the dataloader to move forward
            tune_cfg: quantization config
            alpha: smooth alpha in SmoothQuant, 1.0 will fallback to SPIQ
            folding: whether insert mul(False) or just allow foldable layers(True) for SmoothQuant
            percentile: percentile of calibration to remove outliers
            op_types: The op types whose input tensor will be dumped
            scales_per_op: True, each op will have an individual scale, mainly for accuracy
                           False, ops with the same input will share a scale, mainly for performance

        Returns:
            model: A smoothed Tensorflow model
        """
        # Get the nodes list which can't be quantized from tune_cfg
        black_nodes = []

        # Run calibration to get max values per channel
        from smooth_quant import SmoothQuantCalibration
        calibration = SmoothQuantCalibration(model_path, self.evaluate, self.dst, \
                                                    op_types, percentile, black_nodes)
        max_vals_per_channel, sq_weight_node_names, sq_weight_tensor_dict = calibration()

        # Calculate the smooth quant scaler and insert Mul op into the graph
        from smooth_quant  import SmoothQuantScaler
        scaler = SmoothQuantScaler(model_path, self.dst, alpha, scales_per_op)
        sq_graph_def, self._saved_model, self.func, \
            self.frozen_func, self.sq_weight_scale_dict = scaler.transform(max_vals_per_channel,
                                          sq_weight_tensor_dict, sq_weight_node_names)
        return sq_graph_def

    def __call__(self):
        sq_graph_def = self.smooth_quant(self.src)

        f=tf.io.gfile.GFile('extracted_graph_def.pb','wb')
        f.write(sq_graph_def.SerializeToString()) 

        graph_def = self.inc_preoptimize(sq_graph_def)

        print('Start to apply quantization')
        quantized_graph_def = self.quantize(graph_def)

        f=tf.io.gfile.GFile('converted_graph_def.pb','wb')
        f.write(quantized_graph_def.SerializeToString()) 

        print('Save Quantized model to ', self.dst)
        model = self._adjust_weight(quantized_graph_def)
        graph_def, _saved_model, func, frozen_func = parse_saved_model(model)
        reconstruct_saved_model(graph_def, func, frozen_func, _saved_model, self.dst)