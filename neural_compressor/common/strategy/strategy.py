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

from typing import List, Set

from neural_compressor.common.strategy.sampler import BaseSampler
from neural_compressor.utils import logger


class Strategy:
    def __init__(self, fp32_model, tuning_config) -> None:
        self.sampler_set: Set[BaseSampler] = set()

    def add_sampler(self, sampler: BaseSampler):
        self.sampler_set.add(sampler)

    def need_stop(self, acc) -> bool:
        return False

    def tuning_summary(self):
        """Show tuning summary."""
        logger.info("*" * 40)
        logger.info("There are {} tuning stages in total.".format(len(self.sampler_set)))
        for i, sampler in enumerate(self.sampler_set):
            logger.info("Stage {}: {}.".format(i, sampler.name))
        logger.info("*" * 40)

    def traverse(self, quantizer):
        self.tuning_summary()
        for cur_sampler in self.sampler_set:
            logger.info("Start tuning stage: {}".format(cur_sampler.name))
            for config in cur_sampler:
                q_model = quantizer.internal_quantize(config)
                acc = quantizer.evaluate(q_model)
                if cur_sampler.need_stop(acc):
                    return q_model
            logger.info("*" * 40)
