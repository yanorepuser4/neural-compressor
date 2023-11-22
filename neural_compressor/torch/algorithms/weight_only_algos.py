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


from typing import Dict, Tuple

import torch

from neural_compressor.common.logger import Logger
from neural_compressor.common.utility import GPTQ, RTN_WEIGHT_ONLY_QUANT
from neural_compressor.torch.quantization.config import GPTQConfig, RTNWeightQuantConfig
from neural_compressor.torch.utils import fetch_module, register_algo, set_module

logger = Logger().get_logger()


###################### RTN Algo Entry ##################################
@register_algo(name=RTN_WEIGHT_ONLY_QUANT)
def rtn_quantize_entry(
    model: torch.nn.Module, configs_mapping: Dict[Tuple[str, callable], RTNWeightQuantConfig], *args, **kwargs
) -> torch.nn.Module:
    """The main entry to apply rtn quantization."""
    from neural_compressor.torch.algorithms.rtn import apply_rtn_on_single_module

    for (op_type, op_name), quant_config in configs_mapping.items():
        original_module = fetch_module(model, op_name)
        logger.info(f"Apply RTN on module: {op_name}, {original_module}")
        rtn_module = apply_rtn_on_single_module(original_module, quant_config)
        set_module(model, op_name, rtn_module)
    return model


###################### GPTQ Algo Entry ##################################


@register_algo(name=GPTQ)
def gptq_quantize_entry(
    model: torch.nn.Module, configs_mapping: Dict[Tuple[str, callable], GPTQConfig], dataloader, *args, **kwargs
) -> torch.nn.Module:
    logger.info("quantizing with the GPTQ algorithm")
    from neural_compressor.torch.algorithms.gptq import apply_gptq_quantize, gptq_config_mapping

    weight_config, nsamples, use_max_length, pad_max_length, device = gptq_config_mapping(configs_mapping)
    model, quantization_perm = apply_gptq_quantize(
        model=model,
        weight_config=weight_config,
        dataloader=dataloader,
        nsamples=nsamples,
        use_max_length=use_max_length,
        pad_max_length=pad_max_length,
        device=device,
        layer_wise=False,
        model_path=None,
    )
    # Assign the gptq config as an attribute of model
    model._gptq_quantization_perm = quantization_perm
    return model
