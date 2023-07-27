import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.training import saver
from tensorflow.core.protobuf import config_pb2
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

# from insert_qdq import GenerateGraphWithQDQPattern
from configs import op_wise_config, int8_sequences

class ConvertSavedModel():
    def __init__(self, src='./gpt-j-6B', dst='./converted_gpt-j-6B', quantize=False, evaluate=None):
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
        self.apply_quantize = quantize
        self.op_wise_config = op_wise_config
        self.int8_sequences = int8_sequences

    def _apply_inlining(self, func):
        """Apply an inlining optimization to the function's graph definition."""
        graph_def = func.graph.as_graph_def()

        # In some cases, a secondary implementation of the function (e.g. for GPU) is
        # written to the "api_implements" attribute. (e.g. `tf.keras.layers.LSTM` in
        # TF2 produces a CuDNN-based RNN for GPU).
        # This function suppose to inline all functions calls, but "api_implements"
        # prevents this from happening. Removing the attribute solves the problem.
        # To learn more about "api_implements", see:
        #   tensorflow/core/grappler/optimizers/implementation_selector.h
        for function in graph_def.library.function:
            if "api_implements" in function.attr:
                del function.attr["api_implements"]

        meta_graph = saver.export_meta_graph(graph_def=graph_def, graph=func.graph)

        # Clear the initializer_name for the variables collections, since they are not
        # needed after saved to saved_model.
        for name in [
            "variables", "model_variables", "trainable_variables", "local_variables"
        ]:
            raw_list = []
            for raw in meta_graph.collection_def["variables"].bytes_list.value:
                variable = variable_pb2.VariableDef()
                variable.ParseFromString(raw)
                variable.ClearField("initializer_name")
                raw_list.append(variable.SerializeToString())
            meta_graph.collection_def[name].bytes_list.value[:] = raw_list

        # Add a collection 'train_op' so that Grappler knows the outputs.
        fetch_collection = meta_graph_pb2.CollectionDef()
        for array in func.inputs + func.outputs:
            fetch_collection.node_list.value.append(array.name)
        meta_graph.collection_def["train_op"].CopyFrom(fetch_collection)

        # Initialize RewriterConfig with everything disabled except function inlining.
        config = config_pb2.ConfigProto()
        rewrite_options = config.graph_options.rewrite_options
        rewrite_options.min_graph_nodes = -1  # do not skip small graphs
        rewrite_options.optimizers.append("function")

        new_graph_def = tf_optimizer.OptimizeGraph(config, meta_graph)

        return new_graph_def

    def _construct_function_from_graph_def(self, func, graph_def, frozen_func=None):
        """Rebuild function from graph_def."""
        if frozen_func is None:
            frozen_func = func

        # If a function is converted, then the TF context contains the original
        # function while the converted_graph_def contains the converted function.
        # Remove the original function from the TF context in this case.
        for f in graph_def.library.function:
            while context.context().has_function(f.signature.name):
                context.context().remove_function(f.signature.name)

        captures = {
            c[1].name.split(":")[0]: c[0]
            for c in frozen_func.graph.captures
        }
        new_func = wrap_function.function_from_graph_def(
            graph_def, [tensor.name for tensor in frozen_func.inputs],
            [tensor.name for tensor in frozen_func.outputs], captures)
        new_func.graph.structured_outputs = nest.pack_sequence_as(
            func.graph.structured_outputs, new_func.graph.structured_outputs)
        # new_func._function_type = func.function_type  # pylint: disable=protected-access

        # Copy structured input signature from original function (used during
        # serialization)
        new_func.graph.structured_input_signature = (func.structured_input_signature)

        return new_func

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

    def reconstruct_saved_model(self, converted_graph_def):
        converted_func = self._construct_function_from_graph_def(
        self.func, converted_graph_def, self.frozen_func)

        trackable = self._saved_model
        signatures = {signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: converted_func}
        save.save(trackable, self.dst, signatures, options=None)

    def _inference(self, sampling_graph_def):
        import time
        print('Inference the saved_model and capture outputs to files')
        self.reconstruct_saved_model(sampling_graph_def)
        start = time.time()
        _, _ = self.evaluate(self.dst)
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
              graph_def, self._calibration_data, self.op_wise_config,
              self.fake_quant, self.fp32_ops, self.bf16_ops, self.quantized_node_info,
              self.device, self.performance_only, self.itex_mode).do_transformation()

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

    def __call__(self):
        config = tf.compat.v1.ConfigProto()
        config.use_per_session_threads = 1
        config.inter_op_parallelism_threads = 1

        self._saved_model = load.load(self.src, [tag_constants.SERVING])
        self.func = self._saved_model.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

        inlined_graph_def = self._apply_inlining(self.func)
        # self._annotate_variable_ops(func, inlined_graph_def)
        self.frozen_func = self._construct_function_from_graph_def(self.func, inlined_graph_def)

        frozen_graph_def = self.frozen_func.graph.as_graph_def()
        grappler_meta_graph_def = saver.export_meta_graph(
            graph_def=frozen_graph_def, graph=self.frozen_func.graph)

        # Add a collection 'train_op' so that Grappler knows the outputs.
        fetch_collection = meta_graph_pb2.CollectionDef()
        for array in self.frozen_func.inputs + self.frozen_func.outputs:
            fetch_collection.node_list.value.append(array.name)
            grappler_meta_graph_def.collection_def["train_op"].CopyFrom(
                fetch_collection)

        grappler_session_config = config_pb2.ConfigProto()
        rewrite_options = grappler_session_config.graph_options.rewrite_options
        rewrite_options.min_graph_nodes = -1
        extracted_graph_def = tf_optimizer.OptimizeGraph(grappler_session_config,
                                            grappler_meta_graph_def, graph_id=b"tf_graph")

        f=tf.io.gfile.GFile('extracted_graph_def.pb','wb')
        f.write(extracted_graph_def.SerializeToString()) 

        extracted_graph_def = self.inc_preoptimize(extracted_graph_def)
        print('Start to apply quantization')
        if self.apply_quantize:
            converted_graph_def = self.quantize(extracted_graph_def)
        else:
            converted_graph_def = extracted_graph_def

        f=tf.io.gfile.GFile('converted_graph_def.pb','wb')
        f.write(converted_graph_def.SerializeToString()) 
        print('Save Quantized model to ', self.dst)
        self.reconstruct_saved_model(converted_graph_def)