#
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
"""Block detector for Transformer-based TF model."""

from ...utils.utility import LazyImport

from collections import defaultdict
class Pattern:
    anchor_op_type = None
    output_node_type_lst = []
    """
    Example:
        attention_add =

    """
    def __init__(self, graph_info, anchor_op_type, output_node_type_lst) -> None:
        self.graph_info = graph_info
        self.anchor_op_type = anchor_op_type
        self.output_node_type_lst = output_node_type_lst

    def match(self, node_name):
        if node_name not in self.graph_info:
            return False
        node = self.graph_info[node_name].node
        if node.op != self.anchor_op_type:
            return False
        output_node_lst = [self.graph_info[output_node_name].node for\
            output_node_name in self.graph_info[node_name].outputs ]
        dst_output_node_type_lst = defaultdict(int)
        for node in output_node_lst:
            dst_output_node_type_lst[node.op] += 1
        if dst_output_node_type_lst == self.output_node_type_lst:
            print(f"Matched {dst_output_node_type_lst} with {self.output_node_type_lst}")
            return True


# attention_add = Pattern(graph_info, 'AddV2', {'MatMul', 'MatMul', 'MatMul', 'AddV2'})
# ffn_add = Pattern(graph_info, 'AddV2', {'MatMul', 'AddV2'})



class TransformerBasedModelBlockPatternDetector:
    """Detect the attention block and FFN block in transformer-based model."""
    pass