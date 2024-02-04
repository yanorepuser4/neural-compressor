# Copyright (c) 2024 Intel Corporation
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

from typing import Any, Callable, List, Union


def is_list_of_list(value):
    if not isinstance(value, list):
        return False
    for inner_list in value:
        if not isinstance(inner_list, list):
            return False
    return True


def is_list_of_dict(value):
    if not isinstance(value, list):
        return False
    for inner_dict in value:
        if not isinstance(inner_dict, dict):
            return False
    return True


def is_list(value):
    return isinstance(value, list)


class Parameter:
    name: str
    default_val: Any
    support_val: Union[List[Any], List[Callable]]
    default_tuning_options: List[Any]
    support_tuning_options: List[Any]
    tuning_type: Union[Callable, List[Callable]]
    tunable: bool


op_types_param = Parameter(
    name="op_types",
    default_val=["Conv", "GEMM"],
    support_val=["Conv", "GEMM", "MatMul"],
    default_tuning_options=[["Conv", "GEMM"], ["Conv", "GEMM", "MatMul"]],
    support_tuning_options=[["Conv", "GEMM"], ["Conv", "GEMM", "MatMul"]],
    tuning_type=is_list_of_list,
    tunable=True,
)


def between_0_and_1(value):
    return 0 <= value <= 1


sq_param = Parameter(
    name="alpha",
    default_val=0.5,
    support_val=between_0_and_1,
    default_tuning_options=[0.5, 0.1],
    support_tuning_options=[0.5, 0.5],
    tuning_type=is_list,
    tunable=True,
)
