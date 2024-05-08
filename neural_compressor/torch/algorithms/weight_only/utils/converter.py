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

import torch

from neural_compressor.torch.utils import logger, set_module


class FormatConverter:
    def __init__(self):
        K_dim_pack = {"bits": 4, "dim": 1, "out_dtype": torch.int32, "transpose": True}
        N_dim_pack = {"bits": 4, "dim": 0, "out_dtype": torch.int32, "transpose": True}
        self.format = {
            "auto_gptq": {"qweight": K_dim_pack, "qzeros": N_dim_pack},
            "auto_awq": {"qweight": N_dim_pack, "qzeros": N_dim_pack},
        }

    def convert(self, woq_model, input_format="auto_gptq", output_format="auto_awq"):
        # user interface
        for name, module in woq_model.named_modules():
            logger.info("converting layer:{}".format(name))
            module = self.convert_module(module, input_format, output_format)
            set_module(woq_model, name, module)
        return woq_model

    def convert_module(self, woq_module, input_format="auto_gptq", output_format="auto_awq"):
        param_list = ["qweight", "qzeros"]
        for param_name in param_list:
            packed_tensor = getattr(woq_module, param_name)
            raw_tensor = converter.unpack(packed_tensor, bits=4, dim=1, out_dtype=torch.float)
            packed_tensor = converter.pack(raw_tensor, bits=4, dim=1, out_dtype=torch.int32)
            set_module(woq_module, param_name, packed_tensor)

    def pack(self, int_weight, bits=4, dim=1, out_dtype=torch.int32, transpose=True):
        # import pdb; pdb.set_trace()
        # Validate input_data according to input_format
        # Implement validation logic here
        pass

    def unpack(self, packed_weight, bits=4, dim=1, out_dtype=torch.float, transpose=True):
        # import pdb; pdb.set_trace()
        # Validate input_data according to input_format
        # Implement validation logic here
        pass

    def validate(self, input_data):
        # Validate input_data according to input_format
        # Implement validation logic here
        pass

    def clean(self, input_data):
        # Clean input_data (if required) before conversion
        # Implement data cleaning logic here
        pass

    def convert_file(self, input_file, output_file):
        # Convert data from input_file to output_file
        # Implement file conversion logic here
        pass


# Example usage:
converter = FormatConverter(input_format="auto_gptq", output_format="auto_awq")
input_data = {...}  # Input data in JSON format
converter.validate(input_data)
converter.clean(input_data)
output_data = converter.convert(input_data)
