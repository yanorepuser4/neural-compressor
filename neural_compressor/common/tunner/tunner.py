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

from typing import Set
from neural_compressor.common.config import AccuracyCriterion, TuningCriterion
from neural_compressor.common.objective import MultiObjective
from neural_compressor.common.tunner.sampler import BaseSampler
from neural_compressor.common.tunner.utility import create_objectives
from neural_compressor.common import logger


class Strategy:
    @property
    def accuracy_criterion(self) -> AccuracyCriterion:
        return self._accuracy_criterion

    @accuracy_criterion.setter
    def accuracy_criterion(self, value):
        self._accuracy_criterion = value

    @property
    def tuning_criterion(self) -> TuningCriterion:
        return self._tuning_criterion

    @tuning_criterion.setter
    def tuning_criterion(self, value):
        self._tuning_criterion = value

    @property
    def tuning_objectives(self) -> MultiObjective:
        return self._tuning_objectives

    @tuning_objectives.setter
    def tuning_objectives(self, value):
        self._tuning_objectives = value

    @property
    def adaptor_eval_func(self):
        return self._adaptor_eval_func

    @adaptor_eval_func.setter
    def adaptor_eval_func(self, value):
        self._adaptor_eval_func = value

    @property
    def baseline_model(self):
        return self._baseline_model

    @baseline_model.setter
    def baseline_model(self, model):
        self._baseline_model = model

    def __init__(self, baseline_model, accuracy_criterion, tuning_criterion, eval_func) -> None:
        self.sampler_set: Set[BaseSampler] = set()
        self._baseline_model = baseline_model
        self._accuracy_criterion = accuracy_criterion
        self._tuning_criterion = tuning_criterion
        self._tuning_objectives = None
        self._adaptor_eval_func = eval_func
        self._post_init()

    def _need_tuning(self):
        return self.adaptor_eval_func is not None

    def _post_init(self):
        if self._need_tuning():
            self._post_tuning_init()
        else:
            self._post_no_tuning_init()

    def _post_tuning_init(self):
        # 1. create tuning objectives
        tunner_objectives = create_objectives(self.accuracy_criterion, self.tuning_criterion)
        self.tuning_objectives = tunner_objectives

    def _post_no_tuning_init(self):
        """Post init for no tuning."""
        pass

    def evaluate(self, model):
        """Interface of evaluating model.
        #TODO adaptor needs pass a eval_func to tuner, if it call tuner.
        # It's adaptor's responsibility to create `eval_func` according to the `eval_metric` and `eval_dataloader`.
        Args:
            model (object): The model to be evaluated.

        Returns:
            Objective: The objective value evaluated.
        """
        val = self.tuning_objectives.evaluate(self.adaptor_eval_func, model)
        return val

    def add_sampler(self, sampler: BaseSampler):
        """Add sampler into samplers list."""
        self.sampler_set.add(sampler)

    def tuning_summary(self):
        """Show tuning summary."""
        logger.info("*" * 40)
        logger.info(f"There are {len(self.sampler_set)} tuning stages in total.")
        for i, sampler in enumerate(self.sampler_set):
            logger.info("Stage {}: {}.".format(i, sampler.name))
        logger.info("*" * 40)

    def traverse(self, quantizer):
        self.tuning_summary()
        for cur_sampler in self.sampler_set:
            logger.info("Start tuning stage: {}".format(cur_sampler.name))
            for config in cur_sampler:
                q_model = quantizer.internal_quantize(config)
                acc = self.evaluate(q_model)
                if self.need_stop(acc):
                    return q_model
            logger.info("*" * 40)

    def need_stop(self, acc) -> bool:
        """Determine whether to stop sampling.

        #TODO (Yi, update it/or move it to sampler?)
        1. should we move it into sampler class?
        The different sampler may have different stop criterion.
        quant_level = 0, continue sampling even if the accuracy is not satisfied.
        quant_level = 1 with basic, stop sampling if the accuracy is satisfied.

        # exit policy
        1. not_tuning(performance_only): only quantize the model without tuning or evaluation.
        2. timeout = 0, exit the tuning process once it is found model meets the accuracy requirement.
        3. max_trials, the number of the actually trials is less or equal to the max_trials
        4. quant_level = 0, to seek the better performance, continue the tuning process even if the accuracy is satisfied.
        There are two ways to use max_trials to dominate the exit policy.
        1) timeout = 0, the tuning process exit when the actual_trails_count >= max_trials or
           a quantized model meets the accuracy requirements
        2) timeout = inf, the tuning process exit until the trials_count >= max_trials
        Some use case:
        1) Ending tuning process after a quantized model meets the accuracy requirements
           max_trials = inf, timeout = 0 (by default) # the default max_trials is 100
        value of timeout. max_trials control the exit policy
        2) Even after finding a model that meets the accuracy goal, we may want to continue the
           tuning process for better performance or other objectives.
           timeout = 100000 (seconds), max_trials = 10 # Specifics a fairly large timeout, use max_trials
                                             # to control the exit policy.
        3) Only want to try a certain number of trials
           timeout = 100000 (seconds), max_trials = 3 # only want to try the first 3 trials
        """
        return False
