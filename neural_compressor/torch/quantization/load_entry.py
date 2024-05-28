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

import json
import os

from neural_compressor.common.utils import FP8_QUANT  # unified namespace
from neural_compressor.common.utils import load_config_mapping  # unified namespace
from neural_compressor.torch.quantization.config import (
    AutoRoundConfig,
    AWQConfig,
    FP8Config,
    GPTQConfig,
    RTNConfig,
    TEQConfig,
)

config_name_mapping = {
    FP8_QUANT: FP8Config,
}


def load(model_name_or_path="./saved_results", model=None, format="default", *model_args, **kwargs):
    if format == "default":
        from neural_compressor.common.base_config import ConfigRegistry

        qconfig_file_path = os.path.join(os.path.abspath(os.path.expanduser(model_name_or_path)), "qconfig.json")
        with open(qconfig_file_path, "r") as f:
            per_op_qconfig = json.load(f)

        if " " in per_op_qconfig.keys():  # ipex qconfig format: {' ': {'q_op_infos': {'0': {'op_type': ...
            from neural_compressor.torch.algorithms.static_quant import load

            return load(model_name_or_path)
        else:
            config_mapping = load_config_mapping(qconfig_file_path, ConfigRegistry.get_all_configs()["torch"])
            # select load function
            config_object = config_mapping[next(iter(config_mapping))]
            if isinstance(config_object, (RTNConfig, GPTQConfig, AWQConfig, TEQConfig, AutoRoundConfig)):  # WOQ
                from neural_compressor.torch.algorithms.weight_only.save_load import load

                return load(model_name_or_path, format, *model_args, **kwargs)

            model.qconfig = config_mapping
            if isinstance(config_object, FP8Config):  # FP8
                from neural_compressor.torch.algorithms.habana_fp8 import load

                return load(model, model_name_or_path)  # pylint: disable=E1121
    elif format == "huggingface":
        # now only support load huggingface WOQ model
        from neural_compressor.torch.algorithms.weight_only.save_load import load

        return load(model_name_or_path, format, *model_args, **kwargs)
