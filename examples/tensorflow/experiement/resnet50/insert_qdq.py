import copy
from collections import namedtuple
from tensorflow.python.framework import dtypes
from neural_compressor.adaptor.tf_utils.graph_util import GraphAnalyzer
from neural_compressor.adaptor.tf_utils.graph_rewriter.graph_base import GraphRewriterBase
from neural_compressor.adaptor.tf_utils.graph_util import GraphRewriterHelper as Helper

class InsertQDQPatternBeforeMatmul(GraphRewriterBase):
    """Insert Q/DQ pairs before quantizable ops."""
    def __init__(self, model):
        """Initilization."""
        super().__init__(model)
        self.device = 'cpu'
        self.node_details = namedtuple('node_details', ['node', 'output'])
        self.node_name_mapping = {}
        self.check_op_list = {"MatMul"}

        for node in self.model.node:
            if node.name in self.node_name_mapping:
                raise ValueError("Duplicate Node Found when _parse_graph, the node name is {}" \
                    .format(node.name))
            self.node_name_mapping[node.name] = self.node_details(node=node, output=[])
        for node_name in self.node_name_mapping:
            for each_input in self.node_name_mapping[node_name].node.input:
                self.node_name_mapping \
                    [Helper.node_name_from_input(each_input)].output.append(node_name)

    def do_transformation(self):
        """Generate the graph with QDQ patterns, this is the first step to do new api quantizaiton."""
        self.g = GraphAnalyzer()
        self.g.graph = copy.deepcopy(self.model)
        self.graph_info = self.g.parse_graph()

        self.g.get_frame_info()

        quantizable_op_names = []
        for node in self.model.node:
            if node.op == 'MatMul':
                quantizable_op_names.append(node.name)

        # insert QDQ pattern for op's input
        for op_name in quantizable_op_names:
            self._insert_qdq_pattern_for_common_ops(self.graph_info[op_name].node,
                                                    False)

        return self.g.dump_graph()

    def _insert_qdq_pattern_for_common_ops(self, original_node, is_asymmetric):
        """Insert QDQ patterns for common OPs."""
        namespace_prefix = original_node.name + "_eightbit"

        all_inputs = self.node_name_mapping[original_node.name].node.input[:1]
        for each_input_name in all_inputs:
            if each_input_name[0] == '^':
                continue

            dtype = dtypes.quint8
            self._insert_qdq_pattern_for_each_input(original_node.name,
                                                    namespace_prefix,
                                                    each_input_name,
                                                    is_asymmetric,
                                                    dtype,
                                                    device=self.device)


    def _insert_qdq_pattern_for_each_input(self, op_name, namespace_prefix,
                                           input_name, is_asymmetric,
                                           dtype=dtypes.quint8, input_index=0,
                                           device='cpu'):
        """Takes one float input to an op, and converts it to quantized form."""
        unique_input_name = input_name.replace(":", "__port__").replace("^", "__hat__")
        min_input_name = namespace_prefix + "_min_" + unique_input_name
        max_input_name = namespace_prefix + "_max_" + unique_input_name
        quantize_input_name = namespace_prefix + "_quantize_" + unique_input_name

        reshape_dims_name = namespace_prefix + "_reshape_dims" + unique_input_name
        reduction_dims_name = namespace_prefix + "_reduction_dims" + unique_input_name

        min_node = Helper.create_constant_node(
            min_input_name, -1., dtypes.float32, device="cpu")
        max_node = Helper.create_constant_node(
            max_input_name, 1., dtypes.float32, device="cpu")
        quant_v2_node = Helper.create_node(
            "QuantizeV2", quantize_input_name,
            [input_name, min_input_name, max_input_name])
        Helper.set_attr_dtype(quant_v2_node, "T", dtype)
        if not is_asymmetric:
            Helper.set_attr_string(quant_v2_node, "round_mode", b"HALF_TO_EVEN")
        #Helper.set_attr_bool(quant_v2_node, "narrow_range", False if is_asymmetric else True)
        if "BatchMatMul" in self.graph_info[op_name].node.op:
            Helper.set_attr_string(
                quant_v2_node, "mode", b"SCALED")
        else:
            Helper.set_attr_string(
                quant_v2_node, "mode", b"MIN_FIRST" if is_asymmetric else b"SCALED")

        if "Concat" in self.graph_info[op_name].node.op:
            dequantize_node = Helper.create_node(
                "Dequantize", op_name + '_dequantize_' + str(input_index),
                [quant_v2_node.name, quant_v2_node.name + ':1', quant_v2_node.name + ':2'])
        else:
            dequantize_node = Helper.create_node(
                "Dequantize", op_name + '_dequantize',
                [quant_v2_node.name, quant_v2_node.name + ':1', quant_v2_node.name + ':2'])
        Helper.set_attr_dtype(dequantize_node, "T", dtype)
        if "BatchMatMul" in self.graph_info[op_name].node.op:
            Helper.set_attr_string(
                dequantize_node, "mode", b"SCALED")
        else:
            Helper.set_attr_string(
                dequantize_node, "mode", b"MIN_FIRST" if is_asymmetric else b"SCALED")

        self.g.add_node(quant_v2_node,
                        self.graph_info[op_name].node.input[0],
                        [dequantize_node.name])
        self.g.add_node(dequantize_node, quant_v2_node.name, [op_name])
        self.g.add_node(min_node, None, [quant_v2_node.name])
        self.g.add_node(max_node, None, [quant_v2_node.name])
        self.graph_info[op_name].node.input[input_index] = dequantize_node.name