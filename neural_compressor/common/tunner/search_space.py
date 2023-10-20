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

from enum import Enum
from typing import Any, Dict, List

from neural_compressor.common import logger


class ExpandMode(Enum):
    PRODUCT = 1
    CONCAT = 2


class HyperParams:
    """Hyper parameter group.

    Examples:
        product_demo_hp = HyperParams(
            name="product_demo_hp",
            params_space={
                'a': ['a1', 'a2'],
                'b': ['b1', 'b2']
            },
            expand_mode=ExpandMode.PRODUCT
        )

        for a in ['a1', 'a2']:
            for b in ['b1', 'b2']:
                tune_cfg['a'] = a
                tune_cfg['b'] = b
                yield tune_cfg

        concat_demo_hp = HyperParams(
            name="concat_demo_hp",
            params_space={
                'a': ['a1', 'a2'],
                'b': ['b1', 'b2']
            },
            expand_mode=ExpandMode.CONCAT
        )

        for a in ['a1', 'a2']:
            tune_cfg['a'] = a
            yield tune_cfg
        for b in ['b1', 'b2']:
            tune_cfg['b'] = b
            yield tune_cfg
    """

    def __init__(self, name: str = "", params_space: Dict[str, List[Any]] = None, expand_mode: str = ExpandMode.CONCAT):
        """_summary_

        Args:
            name: the name of the hyper parameter group. Defaults to "".
            params_space: the space of the hyper parameter group. Defaults to None.
            expand_mode: the expand mode, 'product' or 'concat'. Defaults to ExpandMode.CONCAT.
        """
        self.name = name
        self.params_space = params_space
        self.expand_mode = expand_mode

    def exapnd(self):
        """Expand the hyper parameter group."""
        if self.expand_mode == ExpandMode.PRODUCT:
            return self._expand_product()
        elif self.expand_mode == ExpandMode.CONCAT:
            return self._expand_concat()
        else:
            raise NotImplementedError

    def _expand_concat(self):
        """Expand the hyper parameter group by concatenating all the parameters."""
        for k, v in self.params_space.items():
            for i in v:
                yield {k: i}

    def _expand_product(self):
        """Expand the hyper parameter group by product all the parameters."""
        logger.info("Waiting for implementation")
        raise NotImplementedError
