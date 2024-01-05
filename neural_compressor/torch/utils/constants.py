# -*- coding: utf-8 -*-
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


# double quant params

DOUBLE_QUANT_CONFIGS = {
    "GGML_TYPE_Q2_K": {
        "weight_dtype": "int",
        "weight_bits": 2,
        "weight_group_size": 16,
        "weight_sym": False,
        "double_quant_bits": 4,
        "double_quant_dtype": "int",
        "double_quant_sym": False,
        "double_quant_group_size": 16,
    },
    "GGML_TYPE_Q3_K": {
        "weight_dtype": "int",
        "weight_bits": 3,
        "weight_group_size": 16,
        "weight_sym": True,
        "double_quant_bits": 6,
        "double_quant_dtype": "int",
        "double_quant_sym": True,
        "double_quant_group_size": 16,
    },
    "GGML_TYPE_Q4_K": {
        "weight_dtype": "int",
        "weight_bits": 4,
        "weight_group_size": 32,
        "weight_sym": False,
        "double_quant_bits": 6,
        "double_quant_dtype": "int",
        "double_quant_sym": False,
        "double_quant_group_size": 8,
    },
    "GGML_TYPE_Q5_K": {
        "weight_dtype": "int",
        "weight_bits": 5,
        "weight_group_size": 32,
        "weight_sym": False,
        "double_quant_bits": 6,
        "double_quant_dtype": "int",
        "double_quant_sym": False,
        "double_quant_group_size": 8,
    },
    "BNB": {
        "weight_dtype": "nf4",
        "weight_bits": 4,
        "weight_group_size": 32,
        "double_quant_bits": 8,
        "double_quant_dtype": "int",
        "double_quant_sym": False,
        "double_quant_group_size": 256,
    },
}

# mixed precision quant params
'''
LLAMA_FTYPE_MOSTLY_Q2_K   - uses GGML_TYPE_Q4_K for the attention.vw and feed_forward.w2 tensors, GGML_TYPE_Q2_K for the other tensors.
LLAMA_FTYPE_MOSTLY_Q3_K_S - uses GGML_TYPE_Q3_K for all tensors
LLAMA_FTYPE_MOSTLY_Q3_K_M - uses GGML_TYPE_Q4_K for the attention.wv, attention.wo, and feed_forward.w2 tensors, else GGML_TYPE_Q3_K
LLAMA_FTYPE_MOSTLY_Q3_K_L - uses GGML_TYPE_Q5_K for the attention.wv, attention.wo, and feed_forward.w2 tensors, else GGML_TYPE_Q3_K
LLAMA_FTYPE_MOSTLY_Q4_K_S - uses GGML_TYPE_Q4_K for all tensors
'''
OP_NAME_MAPPING = {
    "LLAMA":{
        "attention.wv": ".*self_attn.v_proj.*",
        "attention.wo": ".*self_attn.o_proj.*",
        "feed_forward.w2": ".*mlp.down_proj.*",
    },
}

MIXED_QUANT_CONFIGS = {
    "LLAMA_FTYPE_MOSTLY_Q2_K": {
        "global": "GGML_TYPE_Q2_K",
        "local": {
            "attention.wv": "GGML_TYPE_Q4_K",
            "feed_forward.w2": "GGML_TYPE_Q4_K",
        }
    },
    "LLAMA_FTYPE_MOSTLY_Q3_K_S": {
        "global": "GGML_TYPE_Q3_K"
    },
    "LLAMA_FTYPE_MOSTLY_Q3_K_M": {
        "global": "GGML_TYPE_Q3_K",
        "local": {
             "attention.wv": "GGML_TYPE_Q4_K",
             "attention.wo": "GGML_TYPE_Q4_K",
             "feed_forward.w2": "GGML_TYPE_Q4_K",
        },
    },
    "LLAMA_FTYPE_MOSTLY_Q3_K_L": {
        "global": "GGML_TYPE_Q3_K",
        "local": {
             "attention.wv": "GGML_TYPE_Q5_K",
             "attention.wo": "GGML_TYPE_Q5_K",
             "feed_forward.w2": "GGML_TYPE_Q5_K",
        },
    },
    "LLAMA_FTYPE_MOSTLY_Q4_K_S": {
        "global": "GGML_TYPE_Q4_K",
    },
}
