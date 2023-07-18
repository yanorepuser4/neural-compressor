import re
import copy
import numpy as np
from collections import namedtuple
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util
from tensorflow.core.framework import attr_value_pb2
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
        self.op_wise_config = {}

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

        # insert QDQ pattern for op's weight
        self.g_weight = GraphAnalyzer()
        self.g_weight.graph = self.g.dump_graph()
        self.graph_info = self.g_weight.parse_graph()
        target_nodes = self.g_weight.query_fusion_pattern_nodes(
               [["MatMul"]])
        for i in target_nodes:
            if i[0] not in quantizable_op_names:
                continue
            computational_node_name = i[0]

            computational_node = self.graph_info[computational_node_name].node
            weight_name = computational_node.input[1]
            if re.search(r"\w+:\d+", weight_name):
                weight_node = self.graph_info[weight_name.rsplit(':', 1)[0]].node
            else:
                weight_node = self.graph_info[weight_name].node
            if weight_node.op == 'Enter': # pragma: no cover
                if self.itex_mode:
                    parent_node = self.graph_info[Helper.node_name_from_input(weight_node.input[0])].node
                    if not parent_node.op == 'Const':
                        continue
                    weight_node = parent_node
                else:
                    continue

            if computational_node_name in self.op_wise_config.keys():
                op_wise_cfg = self.op_wise_config[computational_node_name]
                per_channel = op_wise_cfg[0]
                weight_bit = op_wise_cfg[3]
            else:
                per_channel = False
                weight_bit = 7

            min_max_values = []
            self._insert_qdq_pattern_for_weight_node(computational_node,
                                                     weight_node,
                                                     weight_name,
                                                     min_max_values,
                                                     per_channel,
                                                     weight_bit,
                                                     self.device)

        return  self.g_weight.dump_graph()

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

    def _insert_qdq_pattern_for_weight_node(self,
                                            computational_node,
                                            weight_node,
                                            weight_name,
                                            min_max_values,
                                            per_channel,
                                            weight_bit=7.0,
                                            device='cpu'):
        """Insert QDQ pattern for weight node."""
        host_op_type = computational_node.op
        base_name = weight_node.name + "_"

        if weight_node.op == "ReadVariableOp":
            qint8_const_name = base_name + "qint8_readvariable"
        else:
            qint8_const_name = base_name + "qint8_const"
        
        min_name = base_name + "min"
        max_name = base_name + "max"
        epsilon = 1e-4  # Needs to be set empirically if accuracy is not satisfactory
        range_coefficent = 127 / (2 ** weight_bit - 1)
        min_value = 0
        max_value = 0
        insert_reshape = False
        shape_convert = None
        shape_revert = None

        # The weight node of BatchMatMul may have no value
        if 'value' in weight_node.attr and \
           host_op_type in ("Conv2D", "MatMul", "BatchMatMul", "BatchMatMulV2", "Conv3D", \
                            "Conv2DBackpropInput", "Conv3DBackpropInputV2"):
            float_tensor = tensor_util.MakeNdarray(weight_node.attr["value"].tensor)
            if per_channel:
                if host_op_type in ('Conv3D', 'Conv3DBackpropInputV2'):
                    ranges = np.abs(float_tensor).max(axis=(0, 1, 2, 3))
                elif host_op_type in ('Conv2D', 'Conv2DBackpropInput'):
                    ranges = np.abs(float_tensor).max(axis=(0, 1, 2))
                else:
                    ranges = np.abs(float_tensor).max(axis=(0, 1))

                ranges *= range_coefficent
                min_value = -ranges
                max_value = ranges
                # nudging min-max values outside epsilon radius around zero
                ranges[ranges < epsilon] = epsilon
                min_value[np.abs(min_value) < epsilon] = -epsilon
                max_value[np.abs(max_value) < epsilon] = epsilon
                # qint8_tensor = (np.around(float_tensor *127.0/ranges)).astype(np.int8)
            else:
                min_value = np.min(float_tensor)
                max_value = np.max(float_tensor)
                min_value *= range_coefficent
                max_value *= range_coefficent
                min_value = min(min_value, 0.0)
                if min_value == max_value:
                    if abs(min_value) < 0.000001:
                        max_value = min_value + 1.0
                    elif min_value > 0:
                        max_value = 2 * min_value
                    else:
                        max_value = min_value / 2.0
                range_value = np.max(np.abs([min_value, max_value]))
                # qint8_tensor = (np.around(float_tensor * 127.0 / range_value)).astype(np.int8)
                # qint8_tensor = np.clip(qint8_tensor, -127, 127).astype(np.int8)
                min_value = -range_value
                max_value = range_value
        elif weight_node.op == "ReadVariableOp":
            min_value = -1
            max_value = 1
        elif host_op_type == "DepthwiseConv2dNative":
            float_tensor = tensor_util.MakeNdarray(weight_node.attr["value"].tensor)
            # get the max values based on dim 0 and 1 for depthwise conv
            # since, the output channel will be dim 2 * dim 3
            ranges = np.abs(float_tensor).max(axis=(0, 1))
            ranges = ranges.flatten()
            min_value = -ranges
            max_value = ranges
            # nudging min-max values outside epsilon radius around zero
            ranges[ranges < epsilon] = epsilon
            min_value[np.abs(min_value) < epsilon] = -epsilon
            max_value[np.abs(max_value) < epsilon] = epsilon
            # Since output channel will be 1 dim which is dim 2 * dim 3
            # When divide by range, qint8_tensor needs to be 3 dim
            # where, 3rd dim should be same dim of ranges
            a, b, c, d = float_tensor.shape
            # qint8_tensor = (np.around(float_tensor.reshape(a, b, c * d) * 127.0 /
            #                ranges)).astype(np.int8)
            # get the shape back to 4 dim
            # qint8_tensor = qint8_tensor.reshape(a, b, c, d)
            if self.itex_mode and d != 1:
                insert_reshape = True
                shape_convert = [a, b, c * d]
                shape_revert = [a, b, c, d]
        else:
            min_value = np.min(min_max_values[computational_node.name+'__min'])
            max_value = np.max(min_max_values[computational_node.name+'__max'])

        min_node = Helper.create_constant_node(min_name, min_value,
                                                            dtypes.float32, device="cpu")
        max_node = Helper.create_constant_node(max_name, max_value,
                                                            dtypes.float32, device="cpu")
        if "BatchMatMul" in host_op_type and "BatchMatMul" not in weight_node.op:
            min_node.input.append("^" + weight_name)
            max_node.input.append("^" + weight_name)

        min_enter_node = None
        max_enter_node = None
        if insert_reshape:
            reshape_dims_4to3_name = qint8_const_name + "_reshape_dims_4to3_"
            reshape_dims_4to3_node = Helper.create_constant_node(
                reshape_dims_4to3_name, shape_convert, dtypes.int32)
            reshape_4to3_name = qint8_const_name + "_reshape_4to3_"
            reshape_4to3_node = Helper.create_node("Reshape", reshape_4to3_name,
                                                    [weight_node.name, reshape_dims_4to3_name])
            reshape_4to3_node.attr["T"].CopyFrom(
                attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
            quant_node = Helper.create_node(
                    "QuantizeV2", qint8_const_name + '_quant',
                    [reshape_4to3_name, min_name, max_name])
        else:
            quant_node = Helper.create_node(
                "QuantizeV2", qint8_const_name + '_quant',
                [weight_node.name, min_name, max_name])

        dequant_node = Helper.create_node(
            "Dequantize", base_name + '_dequant',
            [quant_node.name, quant_node.name + ':1', quant_node.name + ':2'])
        Helper.set_attr_dtype(quant_node, "T", dtypes.qint8)
        Helper.set_attr_string(quant_node, "mode", b"SCALED")
        Helper.set_attr_string(quant_node, "round_mode", b"HALF_TO_EVEN")
        Helper.set_attr_dtype(dequant_node, "T", dtypes.qint8)
        Helper.set_attr_string(dequant_node, "mode", b"SCALED")
        if per_channel:
            if host_op_type in ('Conv2D', 'Conv2DBackpropInput'):
                Helper.set_attr_int(quant_node, 'axis', 3)
                Helper.set_attr_int(dequant_node, 'axis', 3)
            elif host_op_type in ('Conv3D', 'Conv3DBackpropInputV2'):
                Helper.set_attr_int(quant_node, 'axis', 4)
                Helper.set_attr_int(dequant_node, 'axis', 4)
            elif host_op_type == 'MatMul':
                Helper.set_attr_int(quant_node, 'axis', 1)
                Helper.set_attr_int(dequant_node, 'axis', 1)
            else:
                Helper.set_attr_int(quant_node, 'axis', -1)
                Helper.set_attr_int(dequant_node, 'axis', -1)
        if host_op_type == 'DepthwiseConv2dNative':
            Helper.set_attr_int(quant_node, 'axis', 2)
            Helper.set_attr_int(dequant_node, 'axis', 2)

        if insert_reshape:
            reshape_dims_3to4_name = qint8_const_name + "_reshape_dims_3to4_"
            reshape_dims_3to4_node = Helper.create_constant_node(
                reshape_dims_3to4_name, shape_revert, dtypes.int32)
            reshape_3to4_name = qint8_const_name + "_reshape_3to4_"
            reshape_3to4_node = Helper.create_node("Reshape", reshape_3to4_name,
                                                    [dequant_node.name, reshape_dims_3to4_name])
            reshape_3to4_node.attr["T"].CopyFrom(
                attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
            self.g_weight.add_node(reshape_dims_4to3_node, None, [reshape_4to3_name])
            self.g_weight.add_node(reshape_dims_3to4_node, None, [reshape_3to4_name])
            self.g_weight.add_node(reshape_4to3_node, weight_node.name, [quant_node.name])
            self.g_weight.add_node(quant_node, reshape_4to3_name, [])
            self.g_weight.add_node(min_node, None, [quant_node.name])
            self.g_weight.add_node(max_node, None, [quant_node.name])
            self.g_weight.add_node(dequant_node, quant_node.name, [reshape_3to4_name])
            self.g_weight.add_node(reshape_3to4_node, dequant_node.name, [computational_node.name])
            computational_node.input[1] = reshape_3to4_node.name
        else:
            if computational_node.name in self.g.parent_frame_details and \
               self.g.parent_frame_details[computational_node.name]: # pragma: no cover
                weight_enter_node = Helper.create_node('Enter', \
                                            weight_node.name + '_enter', [weight_node.name])
                Helper.set_attr_string(weight_enter_node, 'frame_name',
                    self.g.parent_frame_details[computational_node.name].attr['frame_name'].s)
                Helper.set_attr_dtype(weight_enter_node, 'T', dtypes.float32)
                Helper.set_attr_bool(weight_enter_node, 'is_constant', True)
                Helper.set_attr_int(weight_enter_node, 'parallel_iterations', \
                    self.g.parent_frame_details[computational_node.name].attr['parallel_iterations'].i)

                min_enter_node = Helper.create_node('Enter', min_name + '_enter', [min_name])
                Helper.set_attr_string(min_enter_node, 'frame_name',
                    self.g.parent_frame_details[computational_node.name].attr['frame_name'].s)
                Helper.set_attr_dtype(min_enter_node, 'T', dtypes.float32)
                Helper.set_attr_bool(min_enter_node, 'is_constant', True)
                Helper.set_attr_int(min_enter_node, 'parallel_iterations', \
                 self.g.parent_frame_details[computational_node.name].attr['parallel_iterations'].i)

                max_enter_node = Helper.create_node('Enter', max_name + '_enter', [max_name])
                Helper.set_attr_string(max_enter_node, 'frame_name',
                    self.g.parent_frame_details[computational_node.name].attr['frame_name'].s)
                Helper.set_attr_dtype(max_enter_node, 'T', dtypes.float32)
                Helper.set_attr_bool(max_enter_node, 'is_constant', True)
                Helper.set_attr_int(max_enter_node, 'parallel_iterations',\
                    self.g.parent_frame_details[computational_node.name].attr['parallel_iterations'].i)

                self.g_weight.add_node(quant_node, weight_name, [])
                self.g_weight.add_node(min_node, None, [min_enter_node.name])
                self.g_weight.add_node(max_node, None, [max_enter_node.name])
                self.g_weight.add_node(min_enter_node, min_node.name, [quant_node.name])
                self.g_weight.add_node(max_enter_node, max_node.name, [quant_node.name])
                self.g_weight.add_node(weight_enter_node, weight_node.name, [quant_node.name])
                quant_node.input[0] = weight_enter_node.name
                quant_node.input[1] = min_enter_node.name
                quant_node.input[2] = max_enter_node.name
                self.g_weight.add_node(quant_node, weight_enter_node.name, [])
                self.g_weight.add_node(dequant_node, quant_node.name, [computational_node.name])
                computational_node.input[1] = dequant_node.name
            else:
                self.g_weight.add_node(quant_node, weight_name, [])
                self.g_weight.add_node(min_node, None, [quant_node.name])
                self.g_weight.add_node(max_node, None, [quant_node.name])
                self.g_weight.add_node(dequant_node, quant_node.name, [computational_node.name])
                computational_node.input[1] = dequant_node.name