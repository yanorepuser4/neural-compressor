import os
import copy
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

from neural_compressor.adaptor.tf_utils.transform_graph.rerange_quantized_concat import RerangeQuantizedConcat
from neural_compressor.adaptor.tf_utils.transform_graph.bias_correction import BiasCorrection
from neural_compressor.adaptor.tf_utils.graph_rewriter.generic.remove_training_nodes import RemoveTrainingNodesOptimizer
from neural_compressor.adaptor.tf_utils.graph_rewriter.generic.strip_unused_nodes import StripUnusedNodesOptimizer
from neural_compressor.adaptor.tf_utils.graph_rewriter.generic.fold_batch_norm import FoldBatchNormNodesOptimizer
from neural_compressor.adaptor.tf_utils.graph_rewriter.generic.fuse_pad_with_conv import FusePadWithConv2DOptimizer
from neural_compressor.adaptor.tf_utils.graph_rewriter.generic.strip_equivalent_nodes import StripEquivalentNodesOptimizer
from neural_compressor.adaptor.tf_utils.graph_rewriter.generic.fuse_pad_with_fp32_conv import FusePadWithFP32Conv2DOptimizer
from neural_compressor.adaptor.tf_utils.graph_rewriter.int8.freeze_value import FreezeValueTransformer
from neural_compressor.adaptor.tf_utils.graph_rewriter.int8.freeze_fake_quant import FreezeFakeQuantOpOptimizer
from neural_compressor.adaptor.tf_utils.graph_rewriter.int8.fuse_conv_requantize import FuseConvRequantizeTransformer
from neural_compressor.adaptor.tf_utils.graph_rewriter.int8.fuse_matmul_requantize import FuseMatMulRequantizeTransformer
from neural_compressor.adaptor.tf_utils.graph_rewriter.int8.fuse_matmul_requantize import FuseMatMulRequantizeDequantizeTransformer
from neural_compressor.adaptor.tf_utils.graph_rewriter.int8.fuse_matmul_requantize import FuseMatMulRequantizeNewAPITransformer
from neural_compressor.adaptor.tf_utils.graph_rewriter.int8.fuse_matmul_requantize import FuseMatMulRequantizeDequantizeNewAPITransformer
from neural_compressor.adaptor.tf_utils.graph_rewriter.int8.fuse_conv_redundant_dequantize import FuseConvRedundantDequantizeTransformer
from neural_compressor.adaptor.tf_utils.graph_rewriter.int8.fuse_matmul_redundant_dequantize import FuseMatMulRedundantDequantizeTransformer
from neural_compressor.adaptor.tf_utils.graph_rewriter.int8.scale_propagation import ScaleProPagationTransformer
from neural_compressor.adaptor.tf_utils.graph_rewriter.bf16.bf16_convert import BF16Convert
from neural_compressor.adaptor.tf_utils.graph_rewriter.int8.post_quantized_op_cse import PostCseOptimizer
from neural_compressor.adaptor.tf_utils.graph_rewriter.int8.post_hostconst_converter import PostHostConstConverter
from neural_compressor.adaptor.tf_utils.graph_rewriter.int8.meta_op_optimizer import MetaInfoChangingMemOpOptimizer
from neural_compressor.adaptor.tf_utils.graph_rewriter.qdq.insert_qdq_pattern import GenerateGraphWithQDQPattern
from neural_compressor.adaptor.tf_utils.graph_rewriter.qdq.share_qdq_y_pattern import ShareQDQForItexYPatternOptimizer
from neural_compressor.adaptor.tf_utils.graph_rewriter.qdq.merge_duplicated_qdq import MergeDuplicatedQDQOptimizer

class ConvertSavedModel():
    def __init__(self, src='./gpt-j-6B', dst='./converted_gpt-j-6B', evaluate=None,
                        op_wise_config={}, int8_sequences={}, signature_names=["serving_default"], apply_smooth_quant=True):
        self.src = src
        self.dst = dst
        self.fp32_ops = []
        self.bf16_ops = []
        self.new_api = True
        self.device ='cpu'
        self.itex_mode = False
        self.fake_quant = False 
        self.evaluate = evaluate
        self.first_signature = True
        self.performance_only = False
        self.target_signature_name = None
        self.op_wise_config = op_wise_config
        self.int8_sequences = int8_sequences
        self.weight_tensor_minmax_dict = {}
        self.signature_names = signature_names
        self.apply_smooth_quant = apply_smooth_quant
        self.tmp_path = '/home/dataset_broad/dataset/users/zehaohua/intermediate_saved_model'

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

    def _adjust_weight(self, graph_def_dict):
        """In-place adjust weight by scale.

        Args:
            scale: smooth scale with the shape (ic,)
            weight_node: reference to the original const weight node
            original_weight: numpy value of the original const weight node
        """
        # scale: (ic,)
        from utils import weight_name_mapping
        reconstruct_saved_model(graph_def_dict, self._saved_model, self.tmp_path)
        model = load.load(self.tmp_path, [tag_constants.SERVING])
        if not self.apply_smooth_quant:
            return model

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

    def _freeze_requantization_ranges(self, additional_data=None):
        """Freeze requantization ranges after doing quantization."""
        self._tmp_graph_def, quantizev2_max = FreezeValueTransformer(
            self._tmp_graph_def,
            self._calibration_data,
            '__max:', device=self.device).do_transformation()
        self._tmp_graph_def, quantizev2_min = FreezeValueTransformer(
            self._tmp_graph_def,
            self._calibration_data,
            '__min:', device=self.device).do_transformation()
        self._tmp_graph_def, requant_min_max= FreezeValueTransformer(
            self._tmp_graph_def,
            self._calibration_data,
            '__requant_min_max',
            tensor_data= additional_data,
            device=self.device,
            ).do_transformation()

    def _fuse_requantize_with_fused_quantized_node(self, graph_def):
        """Fuse the Requantize/Dequantize with fused quantized Ops."""
        if self.fake_quant: # pragma: no cover
            self._tmp_graph_def = FreezeFakeQuantOpOptimizer(
                self._tmp_graph_def).do_transformation()

        self._tmp_graph_def = FuseConvRequantizeTransformer(
            self._tmp_graph_def,
            self.device, self.new_api).do_transformation()

        if not self.fake_quant:
            self._tmp_graph_def = FuseMatMulRequantizeNewAPITransformer(
                self._tmp_graph_def).do_transformation()

            self._tmp_graph_def = FuseMatMulRequantizeDequantizeNewAPITransformer(
                self._tmp_graph_def).do_transformation()

        self._tmp_graph_def = StripUnusedNodesOptimizer(
            self._tmp_graph_def,
            ['attention_mask', 'input_ids'],
            ['Identity', 'Identity_1']).do_transformation()

        input_output_names = ['attention_mask', 'input_ids']+ ['Identity', 'Identity_1']
        self._tmp_graph_def = RemoveTrainingNodesOptimizer(
            self._tmp_graph_def,
            protected_nodes=input_output_names).do_transformation()

        self._tmp_graph_def = FoldBatchNormNodesOptimizer(
            self._tmp_graph_def).do_transformation()

        # if self.performance_only or ('scale_propagation_concat' in self.recipes \
        #      and self.recipes['scale_propagation_concat']):
        #     self._tmp_graph_def = RerangeQuantizedConcat(self._tmp_graph_def,
        #         self.device, performance_only=self.performance_only).do_transformation()

        self._tmp_graph_def = MetaInfoChangingMemOpOptimizer(
            self._tmp_graph_def).do_transformation()

        self._tmp_graph_def = StripEquivalentNodesOptimizer(
            self._tmp_graph_def, ['Identity', 'Identity_1']).do_transformation()

        # self._tmp_graph_def = BiasCorrection(
        #     self._tmp_graph_def, self.model.graph_def, self.new_api).do_transformation()

        self._tmp_graph_def.library.CopyFrom(graph_def.library)

    def _inference(self, sampling_graph_def_dict):
        import time
        print('Inference the saved_model and capture outputs to files')
        model = self._adjust_weight(sampling_graph_def_dict)
        start = time.time()
        self.evaluate(model)
        end = time.time()
        print('Calibration Inference Time: ', end-start)

    def quantize(self, graph_def_dict):
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
        
        graph_def = graph_def_dict[self.target_signature_name][0]
        self.quantized_node_info = OptimizeQDQGraph(graph_def,
                                        ['past_key_values', 'attention_mask', 'input_ids'],
                                        ['Identity_1', 'Identity'],
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
            raise "The quantized_node_info should not be empty!"

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
        graph_def_dict[self.target_signature_name][0] = sampling_graph_def
        with CaptureOutputToFile(tmp_dump_file):
            self._inference(graph_def_dict)
        self._calibration_data = Helper.gen_valid_sampling_log(tmp_dump_file)

        graph_def_dict[self.target_signature_name][0] = graph_def
        del sampling_graph_def
        import gc
        gc.collect()

        # Insert QDQ pattern
        self._tmp_graph_def = GenerateGraphWithQDQPattern(
              graph_def, self._calibration_data, self.op_wise_config, self.fake_quant, 
              self.fp32_ops, self.bf16_ops, self.quantized_node_info, self.device, 
              self.performance_only, self.itex_mode, self.weight_tensor_minmax_dict).do_transformation()

        tf.import_graph_def(self._tmp_graph_def, name='')

        self._tmp_graph_def, exclude_node_names = OptimizeQDQGraph(
                self._tmp_graph_def,
                ['attention_mask', 'input_ids'],
                ['Identity', 'Identity_1'],
                self.op_wise_config,
                self.int8_sequences,
                self.device,
                self.fake_quant,
                self.new_api,
                self.performance_only,
                self.itex_mode,
            ).do_transform()
        self._freeze_requantization_ranges({})
        self._fuse_requantize_with_fused_quantized_node(graph_def)

        tf.import_graph_def(self._tmp_graph_def, name='')

        post_optimize_graph_def = FuseMatMulRedundantDequantizeTransformer(self._tmp_graph_def).do_transformation()
        post_optimize_graph_def.library.CopyFrom(self._tmp_graph_def.library)
        self._tmp_graph_def = post_optimize_graph_def

        post_cse_graph_def = PostCseOptimizer(self._tmp_graph_def).do_transformation()
        post_hostconst_graph_def = PostHostConstConverter(post_cse_graph_def).do_transformation()
        post_hostconst_graph_def.library.CopyFrom(self._tmp_graph_def.library)
        self._tmp_graph_def = post_hostconst_graph_def
        
        graph_def_dict[self.target_signature_name][0] = self._tmp_graph_def
        return graph_def_dict

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

        if self.first_signature:
            # Run calibration to get max values per channel
            from smooth_quant import SmoothQuantCalibration
            calibration = SmoothQuantCalibration(model_path, self.evaluate, op_types, percentile,\
                                                        black_nodes, self.signature_names, self.target_signature_name)
            self.max_vals_per_channel, self.sq_weight_node_names, self.sq_weight_tensor_dict = calibration()

        # Calculate the smooth quant scaler and insert Mul op into the graph
        from smooth_quant import SmoothQuantScaler
        scaler = SmoothQuantScaler(model_path, alpha, scales_per_op, self.signature_names, self.target_signature_name)
        sq_graph_def_dict, self._saved_model, self.sq_weight_scale_dict = scaler.transform(self.max_vals_per_channel,
                                          self.sq_weight_tensor_dict, self.sq_weight_node_names)
        return sq_graph_def_dict

    def __call__(self):
        res_graph_def_dict = {}
        for target_signature_name in self.signature_names:
            self.target_signature_name = target_signature_name

            if self.apply_smooth_quant:
                sq_graph_def_dict = self.smooth_quant(self.src)
            else:
                sq_graph_def_dict, self._saved_model = parse_saved_model(self.model, self.signature_names)

            sq_graph_def = sq_graph_def_dict[self.target_signature_name][0]
            f=tf.io.gfile.GFile('sq_graph_def.pb','wb')
            f.write(sq_graph_def.SerializeToString()) 

            sq_graph_def_dict[self.target_signature_name][0] = self.inc_preoptimize(sq_graph_def)
            
            print('Start to apply quantization')
            quantized_graph_def_dict = self.quantize(sq_graph_def_dict)
            
            f=tf.io.gfile.GFile('converted_graph_def.pb','wb')
            f.write(quantized_graph_def_dict[self.target_signature_name][0].SerializeToString()) 
            
            res_graph_def_dict[self.target_signature_name] = quantized_graph_def_dict[self.target_signature_name][0]
        
            self.first_signature = False

        for target_signature_name in self.signature_names:
            quantized_graph_def_dict[target_signature_name][0] = res_graph_def_dict[target_signature_name]

        print('Save Quantized model to ', self.dst)

        model = self._adjust_weight(quantized_graph_def_dict)

        graph_def_dict, _saved_model = parse_saved_model(model, self.signature_names)
        reconstruct_saved_model(graph_def_dict, _saved_model, self.dst)
