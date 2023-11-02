# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from neural_compressor.common.config import (
    _StaticQuantConfig,
    _SmoothQuantConfig
    )

from typing import Union, Any, List
from enum import Enum, auto
import onnxruntime as ort


class OptimizationLevel(Enum):
    """Optimization level for ORT graph."""

    DISABLED = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    BASIC = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    EXTENDED = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    ALL = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    
class DataType(Enum):
    FLOAT32 = auto()
    FP16 = auto()
    BF16 = auto()
    INT8 = auto()

class OrtQuantMode(Enum):
    QDQ = auto()
    QLinear = auto()


# TODO(Yi) remove it before merge 
# Here is an example to show how to port the 2.x to 3.x.
# https://github.com/intel/neural-compressor/blob/5606eaae1e3554f958413339d503afe954c4a26d/neural_compressor/adaptor/onnxrt.yaml#L36
class StaticQuantConfig(_StaticQuantConfig):
    tunable_params = ['act_dtype', 'act_sym', 'weight_dtype', 'weight_sym']
    def __init__(
        self,
        act_dtype: Union[DataType, List[DataType]] = None,
        act_sym: Union[bool, List[bool]] = None,
        act_granularity: Union[Any, List[Any]] = None,
        act_algorithm: Union[bool, List[bool]] = None,
        weight_dtype: Union[DataType, List[DataType]] = None,
        weight_sym: Union[bool, List[bool]] = None,
        weight_granularity: Union[Any, List[Any]] = None,
        weight_algorithm: Union[bool, List[bool]] = None,
        mode: Union[OrtQuantMode, List[OrtQuantMode]] = None,
        white_list: List[str] = None,
        black_list: List[str] = None,
        ) -> None:
        super().__init__(
            act_dtype = act_dtype,
            act_sym = act_sym,
            act_granularity = act_granularity,
            act_algorithm = act_algorithm,
            weight_dtype = weight_dtype,
            weight_sym = weight_sym,
            weight_granularity = weight_granularity,
            weight_algorithm = weight_algorithm,
            white_list = white_list,
            black_list=black_list,
        )
        self.mode = mode


static_config_common = StaticQuantConfig(
    act_dtype=[DataType.FLOAT32, DataType.INT8],
    act_sym=[True, False],
    weight_dtype=[DataType.INT8],
    weight_sym=[True, False],
    white_list=['Conv2d', 'Linear'])


class SmoothQuantConfig(_SmoothQuantConfig):
    tunable_params = ['alpha', 'folding']
    def __init__(
        self,
        alpha: Union[float, List[float]] = None,
        folding: Union[bool, List[bool]] = None,
        ) -> None:
        super().__init__(alpha=alpha, folding=folding)

sq_config = SmoothQuantConfig(alpha=[0.5], folding=[True, False])