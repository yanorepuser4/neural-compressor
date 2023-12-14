#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tensorflow model calibration process for Smooth Quantization."""

import os
import copy
import logging
import tempfile
import numpy as np
import tensorflow as tf
from collections import OrderedDict, UserDict
from tensorflow.core.framework import graph_pb2, attr_value_pb2
from tensorflow.python.framework import tensor_util
from neural_compressor.utils.utility import CaptureOutputToFile
from neural_compressor.adaptor.tf_utils.graph_util import GraphRewriterHelper as Helper
from neural_compressor.adaptor.tf_utils.quantize_graph_common import QuantizeGraphHelper
from neural_compressor.adaptor.tf_utils.util import iterator_sess_run
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import tag_constants
from neural_compressor.adaptor.tf_utils.graph_util import GraphAnalyzer
from utils import parse_saved_model, reconstruct_saved_model, get_suffix
from utils import node_name_from_input, weight_name_mapping
logger = logging.getLogger("neural_compressor")
debug = bool(logger.level == logging.DEBUG)

class SmoothQuantCalibration:
    """A class for performing smooth quantization calibration on a Tensorflow model.

    Args:
        model (Model): The Tensorflow wrapper model to be calibrated.
        dataloader (DataLoader): The data loader for the calibration dataset.
        iterations (int): The number of iterations to run the calibration process.
        op_types (List[str]): The types of operations to be quantized.
        percentile (float): The percentile of calibration to remove outliers.
        black_nodes (List[str]): A list of node names to be ignored during calibration.
    """
    def __init__(self, model_path, evaluate, op_types, \
                percentile, black_nodes, signature_names, target_signature_name):
        """Initializes a SmoothQuantCalibration object."""
        self.model = model_path
        self.evaluate = evaluate
        self.tmp_path = './intermediate_saved_model'
        # self.iterations = 3
        self.op_types = op_types
        self.percentile = percentile
        self.black_nodes = black_nodes
        self._sq_input_node_names = []
        self.print_node_list = []
        self._sq_output_tensor_dict = {}
        self._sq_weight_tensor_dict = {}
        self._sq_weight_node_names = {} # mapping from its weight node name to the concrete output node name
        self.signature_names = signature_names
        self.target_signature_name = target_signature_name

    def _parse_calibration_logs(self, tmp_dump_file):
        valid_data = []
        with open(tmp_dump_file) as file:
            for i in file.readlines():
                if i.startswith(';'):
                    valid_data.append(i.strip())

        for activation in valid_data:
            activation = activation.split(' ')
            data = []
            activation_name = ''
            per_channel = []
            for idx, s in enumerate(activation):
                if idx == 0:
                    per_channel.append(float(s.rsplit(':')[-1].strip('[')))
                    activation_name = s.rsplit(':')[0][1:-9]
                elif s.find('][') != -1:
                    pairs = [float(i) for i in s.split('][')]
                    per_channel.append(pairs[0])
                    data.append(per_channel)
                    per_channel = [pairs[1]]
                elif s.find(']]') != -1:
                    per_channel.append(float(s.strip(']')))
                    data.append(per_channel)
                else:
                    per_channel.append(float(s))

            if activation_name not in self._sq_output_tensor_dict:
                self._sq_output_tensor_dict[activation_name] = [np.array(data)]
            else:
                self._sq_output_tensor_dict[activation_name].append(np.array(data))

    def _insert_print_for_activation(self, graph_def):
        """Insert print node in the graph to do the calibration."""
        cur_graph = GraphAnalyzer()
        cur_graph.graph = graph_def

        graph_info = cur_graph.parse_graph()
        self.graph_info = graph_info
        for cur_list in self.print_node_list:
            self.pre_node_name = cur_list[0]
            self.post_node_name = cur_list[-1]
            insert_node_pairs = []
            top_node = graph_info[self.pre_node_name].node
            if top_node.op == 'ConcatV2':
                for i in range(top_node.attr['N'].i):
                    insert_node_pairs.append([top_node.input[i], self.post_node_name])
            elif top_node.op in ('BatchMatMul', 'BatchMatMulV2'):
                insert_node_pairs.append([top_node.input[0], self.post_node_name])
                if graph_info[top_node.input[1]].node.op != 'Const':
                    insert_node_pairs.append([top_node.input[1], self.post_node_name])
            elif top_node.op in ('Conv2DBackpropInput', 'Conv3DBackpropInputV2'):
                insert_node_pairs.append([top_node.input[2], self.post_node_name])
            else:
                refresh_pre_node_name = graph_info[self.pre_node_name].node.input[0]
                # Check the Conv2D could be fused with previous Pad or not.
                # If so, we need to update the pre-node name correspondingly.
                refresh_pre_node = graph_info[Helper.node_name_from_input(refresh_pre_node_name)].node
                if refresh_pre_node.op == 'Pad' and top_node.op in ('Conv2D', 'Conv3D'):
                    pad_const_node_name = refresh_pre_node.input[1]
                    pad_const_node = graph_info[pad_const_node_name].node
                    padding_tensor = None
                    if graph_info[pad_const_node_name].node.op != 'Const':
                        if pad_const_node.op == 'DataFormatVecPermute':
                            parent_input_node = graph_info[pad_const_node.input[0]].node
                            if parent_input_node.op == 'Const':
                                padding_tensor = tu.MakeNdarray( \
                                    parent_input_node.attr["value"].tensor).flatten()
                    else:
                        padding_tensor = tu.MakeNdarray(pad_const_node.attr["value"].tensor).flatten()
                    if not any(padding_tensor) or \
                        (any(padding_tensor) and (tf.version.VERSION == '1.15.0-up3' or self.new_api)):
                        insert_node_pairs.append([refresh_pre_node_name, self.post_node_name])
                        refresh_pre_node_name = refresh_pre_node.input[0]

                insert_node_pairs.append([refresh_pre_node_name, self.post_node_name])

            output_names = []
            for node_pair_names in insert_node_pairs:
                for index, each_node_name in enumerate(node_pair_names):
                    name_with_sig = each_node_name
                    node_name_prefix = name_with_sig.replace(":", "__port__").replace("^", "__hat__")
                    reshape_dims_name = node_name_prefix + "_reshape_dims"

                    print_node = Helper.create_node(
                        "Print", node_name_prefix + "_print__{}".format(index),
                        [each_node_name + ':0', each_node_name+':0'])

                    if index == 0:
                        msg = ';{}__print__:'.format(each_node_name)
                        # workround for swish_f32, attribute T is not in the op definition
                        if 'swish_f32' in graph_info[self.pre_node_name].node.name:
                            src_dt=attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum)
                        else:
                            src_dt = graph_info[self.pre_node_name].node.attr["T"]
                    else:
                        break

                    print_node.attr["T"].CopyFrom(src_dt)

                    print_node.attr["message"].s = msg.encode()
                    print_node.attr["first_n"].i = -1
                    print_node.attr["summarize"].i = 102400000

                    attr_u = [dtypes.as_dtype(src_dt.type).as_datatype_enum]
                    print_node.attr["U"].list.CopyFrom(
                        attr_value_pb2.AttrValue.ListValue(type=attr_u))
                    post_node_names = graph_info[Helper.node_name_from_input(each_node_name)].outputs
                    if post_node_names:
                        for post_node_name in post_node_names:
                            post_node = graph_info[post_node_name].node
                            if each_node_name not in post_node.input:
                                continue
                            if post_node.op == 'FusedBatchNormV3' and "_print_identity" not in \
                            graph_info[Helper.node_name_from_input(post_node.name)].node.input[0]:
                                identity_node = Helper.create_node("Identity", post_node.name+'_print_identity',
                                    [graph_info[Helper.node_name_from_input(post_node.name)].node.input[0]])
                                identity_node.attr["T"].CopyFrom(src_dt)
                                cur_graph.add_node(identity_node,
                                                graph_info[Helper.node_name_from_input(post_node.name)].node.input[0],
                                                [post_node.name])
                                identity_node.input.append("^" + print_node.name)
                            else:
                                post_node.input.append("^" + print_node.name)
                        
                        cur_graph.add_node(print_node, each_node_name, [])
                    else:
                        identity_node1 = Helper.create_node(
                            "Identity", print_node.name+'_identity', [print_node.name])
                        identity_node1.attr["T"].CopyFrom(src_dt)
                        cur_graph.add_node(print_node, each_node_name, [identity_node1.name])
                        cur_graph.add_node(identity_node1, print_node.name, [])
                        output_names.append(identity_node1.name)
                        
        return cur_graph.dump_graph()

    def _inference(self, sampling_graph_def):
        import time
        print('Inference the saved_model and capture outputs to files')
        sampling_graph_dict = self.graph_func_dict
        sampling_graph_dict[self.target_signature_name][0] = sampling_graph_def
        reconstruct_saved_model(sampling_graph_dict, self._saved_model, self.tmp_path)
        start = time.time()
        self.evaluate(self.tmp_path)
        end = time.time()
        print('Calibration Inference Time: ', end-start)
        self.graph_func_dict[self.target_signature_name][0] = self.graph_def

    def _inference_for_calibration(self):
        """Run the calibration on the input graph.

        Args:
            model(TensorflowBaseModel): input TensorflowBaseModel
        """
        # ITEX optimization has broken INC calibration process.
        # INC needs turn off ITEX optimization pass in calibration stage.
        # TODO ITEX will provide API to replace setting environment variable.
        sampling_graph_def = copy.deepcopy(self.graph_def)
        sampling_graph_def = self._insert_print_for_activation(sampling_graph_def)
        tmp_dump_file = tempfile.mkstemp(suffix='.log')[1]
        print('Start to do calibration for smooth quant')
        with CaptureOutputToFile(tmp_dump_file):
            self._inference(sampling_graph_def)
        self._parse_calibration_logs(tmp_dump_file)
        del sampling_graph_def

    def _get_weight_tensors(self):
        model = load.load(self.model, [tag_constants.SERVING])
        for weight_tensor in model.variables:
            parsed_name = weight_name_mapping(weight_tensor.name)
            if parsed_name in self._sq_weight_node_names:
                self._sq_weight_tensor_dict[parsed_name] = weight_tensor.numpy()

        assert len(self._sq_weight_tensor_dict) == len(self._sq_weight_node_names), \
            'Failed to get weights for some nodes, please check variables'

    def _generate_calibration_data(self):
        """Generate the calibration data."""
        sorted_graph = QuantizeGraphHelper().get_sorted_graph(
            self.graph_def,
            ['attention_mask', 'input_ids'],
            ['Identity', 'Identity_1'],)

        for node in sorted_graph.node:
            if node.op not in self.op_types or node.name in self.black_nodes:
                continue
            # Fix retval already been set issue
            if 'while' in node.input[0]: # pragma: no cover
                continue
            self._sq_input_node_names.append(node.input[0])
            self.print_node_list.append([node.name])
            self._sq_weight_node_names[node.input[1]] = node.name
        self._get_weight_tensors()
        self._inference_for_calibration()

    def _get_maxval_per_channel(self, tensor_data, percentile):
        """Get the max values per input channel.

        Args:
            tensor_data: The input tensors
            percentile: The percentile of calibration to remove outliers

        Returns:
            The max values per input tensor
        """
        permute_datas = []
        for data in tensor_data:    # iteration_num * (N, H, W, C)
            if len(data.shape) == 3:  # pragma: no cover
                # TODO  matmul batchsize*seq*inchannel
                tensor = np.abs(np.reshape(data, (-1, data.shape[-1])))
                permute_datas.append(tensor)
            elif len(data.shape) == 4: # already NHWC
                # tensor = np.transpose(data, [0, 3, 1, 2])
                tensor = data
                tensor = np.abs(np.reshape(tensor, (-1, tensor.shape[-1])))
                permute_datas.append(tensor)
            elif len(data.shape) == 2:  # (?, ic)
                permute_datas.append(np.abs(data))
            else:   # pragma: no cover
                assert False, "not supported"
        permute_datas = np.concatenate(permute_datas, axis=0)
        permute_datas = permute_datas.reshape(-1, permute_datas.shape[-1])
        max_per_channels = np.percentile(permute_datas, percentile, axis=0)
        max_per_channels = max_per_channels.astype(np.single)
        return max_per_channels

    def __call__(self):
        """Generates calibration data and calculate the maximum values per channel.

        Returns:
            max_vals_per_channel (dict): A dictionary containing the maximum values per channel.
            shape_infos (dict): A dictionary containing the shape information.
        """
        # self.graph_def, self._saved_model, self.func, self.frozen_func = parse_saved_model(self.model, )
        self.graph_func_dict, self._saved_model = parse_saved_model(self.model, self.signature_names)
        self.graph_def = self.graph_func_dict[self.target_signature_name][0]
        self._generate_calibration_data()
        max_vals_per_channel = {}
        for key in self._sq_output_tensor_dict.keys():
            max_val_per_channel = self._get_maxval_per_channel(
                self._sq_output_tensor_dict[key], percentile=self.percentile)
            max_vals_per_channel[key] = max_val_per_channel
        return max_vals_per_channel, self._sq_weight_node_names, self._sq_weight_tensor_dict

class SmoothQuantScaler:
    """A class for scaling model weights using Smooth Quantization method.
    
    Args:
        model: Tensorflow model to be scaled
        dataloader: Tensorflow dataloader for the dataset
        alpha: float, the scaling factor
        scales_per_op: bool, each op will have an individual scale or
                       ops with the same input will share a scale
    """

    def __init__(self, model_path, alpha, scales_per_op, signature_names, target_signature_name):
        """Initialization."""
        self.model = model_path
        self.tmp_path = './intermediate_saved_model'
        self.alpha = alpha
        self.scales_per_op = scales_per_op
        self.mul_list = []
        self.signature_names = signature_names
        self.target_signature_name = target_signature_name

    def _adjust_activation(self, scale, input_node_name, output_node_name, w_i):
        """Insert the Mul node after the activation before the weight node.

        Args:
            scale: smooth scale with the shape (ic,)
            input_node_name: the parent input node
            output_node_name: the concrete output weight node name
            w_i: distinguish between different output weight nodes on different branches when naming
        """
        node_suffix = str(w_i)
        mul_const_node = Helper.create_constant_node(input_node_name + "/scale_mul" + node_suffix, scale, tf.float32)
        mul_node = Helper.create_node('Mul', input_node_name + "_mul" + node_suffix,
                            [input_node_name + "/scale_mul" + node_suffix, input_node_name])
        Helper.set_attr_dtype(mul_node, "T", dtypes.float32)
        self.mul_list.append(mul_node.name)
        self.g_analyzer.add_node(mul_node, input_node_name, [output_node_name])
        self.g_analyzer.add_node(mul_const_node, None, [input_node_name + "_mul" + node_suffix])

    def _parse_weight_dict(self, max_vals_per_channel, 
                            sq_weight_tensor_dict, op_types=['MatMul', 'Conv2D']):
        sq_weight_tensors = {}
        sq_weights_node_names = {}
        for input_node_name in max_vals_per_channel:
            curr_weight_tensors = []
            curr_weights_node_names = []
            next_node_names = self.graph_info[input_node_name].outputs
            for node_name in next_node_names:
                curr_node = self.graph_info[node_name].node
                if curr_node.op not in op_types:
                    continue
                if len(curr_node.input) >= 2:
                    weight_name = curr_node.input[1]
                    weight_tensor = sq_weight_tensor_dict[weight_name]
                    curr_weight_tensors.append(weight_tensor)
                    curr_weights_node_names.append(weight_name)
            sq_weight_tensors[input_node_name] = curr_weight_tensors
            sq_weights_node_names[input_node_name] = curr_weights_node_names
        return sq_weight_tensors, sq_weights_node_names


    def transform(self, max_vals_per_channel, sq_weight_tensor_dict, sq_weight_node_names):
        """Apply scaling to weights and activations based on the maximum values per channel.

        Args:
            max_vals_per_channel (dict): A dictionary containing the maximum values per channel for each input node.
            sq_weight_tensors (dict): A dictionary containing the weight tensors for each input node.
            sq_weights_nodes (dict): A dictionary containing the constant nodes for each input node.
            sq_weight_node_names (dict): A dictionary from weight node name to the its concrete output node name.
        
        Returns:
            tuple: A tuple containing the modified model and a list of the inserted multiplication nodes.
        """
        self.graph_def_dict, self._saved_model = parse_saved_model(self.model, self.signature_names)
        self.graph_def = self.graph_def_dict[self.target_signature_name][0]
        self.g_analyzer = GraphAnalyzer()
        self.g_analyzer.graph = self.graph_def
        self.graph_info = self.g_analyzer.parse_graph()
        sq_weight_tensors, sq_weights_node_names = self._parse_weight_dict(max_vals_per_channel, 
                                                            sq_weight_tensor_dict)
        logger.info("Start scaling on model graph for Smooth Quantization.")
        if self.scales_per_op:
            # 1. obtain the smooth scale per op
            # 2. adjust weight
            # 3. adjust activation
            self.sq_weight_scale_dict = {}
            for idx, input_node_name in enumerate(max_vals_per_channel):
                A_max_per_in_channel = max_vals_per_channel[input_node_name]
                W_lst = sq_weight_tensors[input_node_name]  # VQK weight value
                # Use the const nodes before to get weight values, VQK ReadVariable
                W_node_name_lst = sq_weights_node_names[input_node_name]
                # W_node_lst = sq_node_names[input_node_name]
                # Get the concrete weight node as the output of Mul insertion, QKV ReadVariable
                for w_i, W in enumerate(W_lst):
                    if len(W.shape) == 4:
                        # https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
                        # weight: [filter_height, filter_width, in_channels, out_channels]
                        # activation: NHWC, also batch_shape + [in_height, in_width, in_channels]
                        tensor = np.abs(np.transpose(W, [0, 1, 3, 2]))
                        # reduce weight max to (in_channel, ), aligned with activation max
                        W_max_per_in_channel = np.max(np.reshape(tensor, (-1, tensor.shape[-1])), axis=0)
                    elif len(W.shape) == 2: # matmul
                        # reduce weight max to (in_channel, ), aligned with activation max
                        tensor = np.abs(W)
                        W_max_per_in_channel = np.max(tensor, axis=1)
                    else: # pragma: no cover
                        assert False, "not supported"
                    cur_weight_node_name = W_node_name_lst[w_i]
                    try:
                        scale = np.power(A_max_per_in_channel, self.alpha) /  \
                                np.power(W_max_per_in_channel, (1-self.alpha))
                    except ValueError as e: # pragma: no cover
                        logger.info(e)
                        logger.info("Skip smoothing the node: {}".format(cur_weight_node_name))
                        continue
                    # clip the scales that are too small
                    scale = np.clip(scale, a_min=1e-5, a_max=1e8)
                    # skip smoothing the op where scale has elements that less than 1
                    # if np.any(scale < 1):
                    #     logger.info("skip smooth quant: {}".format(input_node_name))
                    #     continue
                    self.sq_weight_scale_dict[cur_weight_node_name] = scale
                    self._adjust_activation(1 / scale, input_node_name, sq_weight_node_names[cur_weight_node_name], w_i)
        else:
            pass
        sq_graph_def = self.g_analyzer.dump_graph()
        sq_graph_def.library.CopyFrom(self.graph_def.library)
        self.graph_def_dict[self.target_signature_name][0] = sq_graph_def
        return  self.graph_def_dict, self._saved_model, self.sq_weight_scale_dict
        